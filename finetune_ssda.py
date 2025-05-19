import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import MinkowskiEngine as ME

from utils.common.check_ddp import is_cuda_available
from utils.common.utils import set_save_dir, check_intensity_params
from utils.datasets.initialization import get_dataset
from configs import get_config
from utils.collation import CollateFN, CollateSSDA
from utils.utils import build_model
from utils.pipelines import PLTTrainer
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


def train(config, args):
    # Check intensity params
    check_intensity_params(args)

    # Parse postprocess_labels
    postprocess_labels = None
    if args.postprocess_labels is not None:
        postprocess_labels = [int(label.strip()) for label in args.postprocess_labels.split(',')]
        print(f"Applying intensity postprocessing to labels: {postprocess_labels}")

    def get_dataloader(dataset, batch_size, collate_fn, shuffle=False, pin_memory=True):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          shuffle=shuffle,
                          num_workers=config.pipeline.dataloader.num_workers,
                          pin_memory=pin_memory)
    try:
        source_mapping_path = config.source_dataset.mapping_path
    except AttributeError('--> Setting default class mapping path for source!'):
        source_mapping_path = None

    try:
        target_mapping_path = config.target_dataset.mapping_path
    except AttributeError('--> Setting default class mapping path for target!'):
        target_mapping_path = None

    source_training_dataset, source_validation_dataset, _ = get_dataset(dataset_name=config.source_dataset.name,
                                                                        dataset_path=config.source_dataset.dataset_path,
                                                                        target_name=config.target_dataset.name,
                                                                        target_path=os.path.dirname(
                                                                            config.target_dataset.dataset_path),
                                                                        voxel_size=config.source_dataset.voxel_size,
                                                                        augment_data=config.source_dataset.augment_data,
                                                                        version=config.source_dataset.version,
                                                                        sub_num=config.source_dataset.num_pts,
                                                                        num_classes=config.model.out_classes,
                                                                        ignore_label=config.source_dataset.ignore_label,
                                                                        weights_path=config.source_dataset.weights_path,
                                                                        mapping_path=source_mapping_path,
                                                                        use_intensity=args.use_intensity != "none",
                                                                        use_traffic_sign_postprocessing=args.use_intensity_postprocess == "true",
                                                                        postprocess_labels=postprocess_labels,
                                                                        postprocess_params_path=args.postprocess_params_path,
                                                                        intensity_path=args.custom_intensity_path,
                                                                        feat_keys=config.model.features)

    target_training_dataset, target_validation_dataset, _ = get_dataset(dataset_name=config.target_dataset.name,
                                                                        dataset_path=config.target_dataset.dataset_path,
                                                                        voxel_size=config.target_dataset.voxel_size,
                                                                        augment_data=config.target_dataset.augment_data,
                                                                        version=config.target_dataset.version,
                                                                        sub_num=config.target_dataset.num_pts,
                                                                        num_classes=config.model.out_classes,
                                                                        ignore_label=config.target_dataset.ignore_label,
                                                                        mapping_path=target_mapping_path,
                                                                        use_intensity=args.use_intensity != "none",
                                                                        use_traffic_sign_postprocessing=args.use_intensity_postprocess == "true",
                                                                        postprocess_labels=postprocess_labels,
                                                                        postprocess_params_path=args.postprocess_params_path,
                                                                        intensity_path=args.custom_intensity_path,
                                                                        feat_keys=config.model.features)

    ssda_sampler = SupervisedSampler(dataset=target_training_dataset,
                                     method=config.adaptation.ssda_sampler.method,
                                     num_frames=config.adaptation.ssda_sampler.num_frames)

    collation = CollateFN()
    training_dataloader = get_dataloader(source_training_dataset,
                                         collate_fn=CollateSSDA(ssda_sampler=ssda_sampler),
                                         batch_size=config.pipeline.dataloader.train_batch_size,
                                         shuffle=True)

    validation_dataloader = get_dataloader(source_validation_dataset,
                                           collate_fn=collation,
                                           batch_size=config.pipeline.dataloader.train_batch_size*4,
                                           shuffle=False)

    model = build_model(config.model)
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    print(f"Model created successfully. {type(model)}")

    if args.checkpoint_path != "":
        config.adaptation.student_checkpoint = args.checkpoint_path
        config.adaptation.teacher_checkpoint = args.checkpoint_path
        print(f"Checkpoints replaced to {args.checkpoint_path}")

    if config.adaptation.teacher_checkpoint:
        model = load_model(config.adaptation.teacher_checkpoint, model)
        print(f"--> Model loaded from checkpoint {config.adaptation.teacher_checkpoint}")
    else:
        raise ValueError("Pre-trained model needed for adaptation, check config.adaptation.teacher_checkpoint")

    pl_module = PLTTrainer(training_dataset=source_training_dataset,
                           validation_dataset=source_validation_dataset,
                           model=model,
                           criterion=config.adaptation.losses.source_criterion,
                           optimizer_name=config.pipeline.optimizer.name,
                           batch_size=config.pipeline.dataloader.train_batch_size,
                           val_batch_size=config.pipeline.dataloader.train_batch_size*4,
                           lr=config.pipeline.optimizer.lr,
                           num_classes=config.model.out_classes,
                           train_num_workers=config.pipeline.dataloader.num_workers,
                           val_num_workers=config.pipeline.dataloader.num_workers,
                           clear_cache_int=config.pipeline.lightning.clear_cache_int,
                           scheduler_name=config.pipeline.scheduler.name)


    save_dir, run_name = set_save_dir(config, args, postprocess_labels=postprocess_labels)
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
                val_dataloaders=validation_dataloader)


if __name__ == '__main__':
    args = parser.parse_args()

    config = get_config(args.config_file)

    # fix random seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True

    train(config, args)
