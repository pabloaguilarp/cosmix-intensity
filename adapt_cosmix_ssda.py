import argparse
import os

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from configs import get_config
from utils.common.check_ddp import is_cuda_available
from utils.common.momentum import MomentumUpdater
from utils.common.utils import check_intensity_params, load_data_concat, load_net_model, set_save_dir
from utils.pipelines.masked_ssda_pipeline import SimMaskedSemiAdaptation
from utils.sampling.ssda import SupervisedSampler

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/source/synlidar_semantickitti.yaml",
                    type=str,
                    help="Path to config file")
parser.add_argument("--checkpoint_path",
                    default="",
                    type=str,
                    help="Path to checkpoint file, for both teacher and student")
parser.add_argument("--use_intensity",
                    type=str,
                    choices=["none", "default", "custom"],
                    help="Intensity mode")
parser.add_argument("--custom_intensity_path",
                    type=str,
                    help="Path to custom intensity predictions")
parser.add_argument("--use_intensity_postprocess",
                    type=str,
                    choices=["true", "false"],
                    default="false",
                    help="Apply intensity random sampling postprocess")
parser.add_argument("--postprocess_labels",
                    type=str,
                    default=None,
                    help="Comma-separated list of labels to apply intensity postprocessing to (e.g., '18,19,20')")
parser.add_argument("--postprocess_params_path",
                    type=str,
                    default="_resources/intensity_postprocess.yaml",
                    help="Path to YAML file with postprocessing parameters")
parser.add_argument("--num_frames",
                    type=int,
                    default=2,
                    help="Number of target frames to include in adaptation process")


def load_model(checkpoint_path, model):
    # reloads model
    def clean_state_dict(state):
        # clean state dict from names of PL
        for k in list(ckpt.keys()):
            if "model" in k:
                ckpt[k.replace("model.", "")] = ckpt[k]
            del ckpt[k]
        return state

    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
    ckpt = clean_state_dict(ckpt)
    model.load_state_dict(ckpt, strict=True)
    return model


def adapt(config, args):
    # Check intensity params
    check_intensity_params(args)

    # Parse postprocess_labels
    postprocess_labels = None
    if args.postprocess_labels is not None:
        postprocess_labels = [int(label.strip()) for label in args.postprocess_labels.split(',')]
        print(f"Applying intensity postprocessing to labels: {postprocess_labels}")

    training_dataset, source_validation_dataset, target_validation_dataset, training_dataloader, validation_dataloaders, target_training_dataset = \
        load_data_concat(args, config, postprocess_labels=postprocess_labels)

    config.adaptation.ssda_sampler.num_frames = args.num_frames
    print(f"Overriding 'num_frames' value, using {config.adaptation.ssda_sampler.num_frames}")

    ssda_sampler = SupervisedSampler(dataset=target_training_dataset,
                                     method=config.adaptation.ssda_sampler.method,
                                     num_frames=config.adaptation.ssda_sampler.num_frames)

    if args.checkpoint_path != "":
        config.adaptation.student_checkpoint = args.checkpoint_path
        config.adaptation.teacher_checkpoint = args.checkpoint_path
        print(f"Checkpoints replaced to {args.checkpoint_path}")

    student_model = load_net_model(config.model, config.adaptation.student_checkpoint, "student")
    teacher_model = load_net_model(config.model, config.adaptation.teacher_checkpoint, "teacher")

    momentum_updater = MomentumUpdater(base_tau=0.999, final_tau=0.999)

    if config.adaptation.self_paced:
        target_confidence_th = np.linspace(config.adaptation.target_confidence_th, 0.6, config.pipeline.epochs)
    else:
        target_confidence_th = config.adaptation.target_confidence_th

    pl_module = SimMaskedSemiAdaptation(training_dataset=training_dataset,
                                        source_validation_dataset=source_validation_dataset,
                                        target_validation_dataset=target_validation_dataset,
                                        student_model=student_model,
                                        teacher_model=teacher_model,
                                        momentum_updater=momentum_updater,
                                        source_criterion=config.adaptation.losses.source_criterion,
                                        target_criterion=config.adaptation.losses.target_criterion,
                                        other_criterion=config.adaptation.losses.other_criterion,
                                        source_weight=config.adaptation.losses.source_weight,
                                        target_weight=config.adaptation.losses.target_weight,
                                        other_weight=config.adaptation.losses.other_weight,
                                        filtering=config.adaptation.filtering,
                                        adaptive_weight=config.adaptation.adaptive_weight,
                                        optimizer_name=config.pipeline.optimizer.name,
                                        train_batch_size=config.pipeline.dataloader.train_batch_size,
                                        val_batch_size=config.pipeline.dataloader.val_batch_size,
                                        lr=config.pipeline.optimizer.lr,
                                        num_classes=config.model.out_classes,
                                        num_workers=config.pipeline.dataloader.num_workers,
                                        clear_cache_int=config.pipeline.lightning.clear_cache_int,
                                        scheduler_name=config.pipeline.scheduler.name,
                                        update_every=config.adaptation.momentum.update_every,
                                        shuffle_batches=config.adaptation.shuffle_batches,
                                        source_filtering=config.adaptation.source_filtering,
                                        weighted_sampling=config.adaptation.weighted_sampling,
                                        target_confidence_th=target_confidence_th,
                                        self_paced=config.adaptation.self_paced,
                                        selection_perc=config.adaptation.selection_perc,
                                        supervised_sampler=ssda_sampler)

    save_dir, run_name = set_save_dir(config, args, num_frames=config.adaptation.ssda_sampler.num_frames, postprocess_labels=postprocess_labels)
    print(f"save_dir: {save_dir}")

    wandb_logger = WandbLogger(project=config.pipeline.wandb.project_name,
                               entity=config.pipeline.wandb.entity_name,
                               name=run_name,
                               offline=config.pipeline.wandb.offline)

    loggers = [wandb_logger]

    # Save only one file (the best checkpoint)
    checkpoint_callback = [ModelCheckpoint(
        dirpath=os.path.join(str(save_dir), 'checkpoints'),
        filename=str(config.model.name).lower(),
        save_top_k=1)]

    if not is_cuda_available():
        trainer = Trainer(max_epochs=config.pipeline.epochs,
                          accelerator='cpu',
                          devices=1,
                          default_root_dir=config.pipeline.save_dir,
                          precision=config.pipeline.precision,
                          logger=loggers,
                          check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                          val_check_interval=1.0,
                          num_sanity_val_steps=config.pipeline.lightning.num_sanity_val_steps,
                          callbacks=checkpoint_callback)
    else:
        trainer = Trainer(max_epochs=config.pipeline.epochs,
                          devices="auto",
                          accelerator="auto",
                          default_root_dir=config.pipeline.save_dir,
                          precision=config.pipeline.precision,
                          logger=loggers,
                          check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                          val_check_interval=1.0,
                          num_sanity_val_steps=config.pipeline.lightning.num_sanity_val_steps,
                          callbacks=checkpoint_callback)

    trainer.fit(pl_module,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloaders)


if __name__ == '__main__':
    args = parser.parse_args()

    config = get_config(args.config_file)

    # fix random seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True

    adapt(config, args)
