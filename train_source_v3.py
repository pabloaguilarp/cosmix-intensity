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
from utils.datasets.initialization import get_dataset
from configs import get_config
from utils.collation import CollateFN
from utils.pipelines import PLTTrainer

from utils.common.utils import check_intensity_params, set_save_dir
from utils.utils import build_model


parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/source/synlidar2semantickitti.yaml",
                    type=str,
                    help="Path to config file")
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


def train(config, args):
    collation = CollateFN()
    def get_dataloader(dataset, batch_size, collate_fn=collation, shuffle=False, pin_memory=True):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          shuffle=shuffle,
                          num_workers=config.pipeline.dataloader.num_workers,
                          pin_memory=pin_memory)
    try:
        mapping_path = config.dataset.mapping_path
    except AttributeError('--> Setting default class mapping path!'):
        mapping_path = None

    # Check intensity params
    check_intensity_params(args)

    # Parse postprocess_labels
    postprocess_labels = None
    if args.postprocess_labels is not None:
        postprocess_labels = [int(label.strip()) for label in args.postprocess_labels.split(',')]
        print(f"Applying intensity postprocessing to labels: {postprocess_labels}")

    training_dataset, validation_dataset, target_dataset = get_dataset(dataset_name=config.dataset.name,
                                                                       dataset_path=config.dataset.dataset_path,
                                                                       target_name=config.dataset.target,
                                                                       target_path=config.dataset.target_path,
                                                                       voxel_size=config.dataset.voxel_size,
                                                                       augment_data=config.dataset.augment_data,
                                                                       version=config.dataset.version,
                                                                       sub_num=config.dataset.num_pts,
                                                                       num_classes=config.model.out_classes,
                                                                       ignore_label=config.dataset.ignore_label,
                                                                       mapping_path=mapping_path,
                                                                       use_intensity=args.use_intensity != "none",
                                                                       use_traffic_sign_postprocessing=args.use_intensity_postprocess == "true",
                                                                       postprocess_labels=postprocess_labels,
                                                                       postprocess_params_path=args.postprocess_params_path,
                                                                       feat_keys=config.model.features)

    print(f"Datasets loaded successfully: {len(training_dataset)} training samples loaded, "
          f"{len(validation_dataset)} validation samples loaded, {len(target_dataset)} target samples loaded")

    training_dataloader = get_dataloader(training_dataset,
                                         collate_fn=collation,
                                         batch_size=config.pipeline.dataloader.batch_size,
                                         shuffle=True)

    validation_dataloader = get_dataloader(validation_dataset,
                                           collate_fn=collation,
                                           batch_size=config.pipeline.dataloader.batch_size*4,
                                           shuffle=False)

    print(f"Dataloaders created successfully")

    model = build_model(config.model)
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    print(f"Model created successfully. {type(model)}")

    pl_module = PLTTrainer(training_dataset=training_dataset,
                           validation_dataset=validation_dataset,
                           model=model,
                           criterion=config.pipeline.loss,
                           optimizer_name=config.pipeline.optimizer.name,
                           batch_size=config.pipeline.dataloader.batch_size,
                           val_batch_size=config.pipeline.dataloader.batch_size*4,
                           lr=config.pipeline.optimizer.lr,
                           num_classes=config.model.out_classes,
                           train_num_workers=config.pipeline.dataloader.num_workers,
                           val_num_workers=config.pipeline.dataloader.num_workers,
                           clear_cache_int=config.pipeline.lightning.clear_cache_int,
                           scheduler_name=config.pipeline.scheduler.name,
                           coords_name="coordinates",
                           features_name="features",
                           labels_name="labels")

    print(f"Lightning Model loaded successfully")

    save_dir, run_name = set_save_dir(config, args, postprocess_labels=postprocess_labels)
    print(f"save_dir: {save_dir}")

    wandb_logger = WandbLogger(project=config.pipeline.wandb.project_name,
                               entity=config.pipeline.wandb.entity_name,
                               name=run_name,
                               offline=config.pipeline.wandb.offline)

    loggers = [wandb_logger]

    checkpoint_callback = [ModelCheckpoint(
        dirpath=os.path.join(str(save_dir), 'checkpoints'),
        filename=str(config.model.name).lower(),
        save_top_k=1)]

    print(f"wandb logger created successfully")

    if not is_cuda_available():
        print("Using CPU because CUDA is not available")
        trainer = Trainer(max_epochs=config.pipeline.epochs,
                          accelerator="cpu",
                          devices=1,
                          default_root_dir=config.pipeline.save_dir,
                          precision=config.pipeline.precision,
                          logger=loggers,
                          check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                          val_check_interval=1.0,
                          num_sanity_val_steps=config.pipeline.lightning.num_sanity_val_steps,
                          log_every_n_steps=config.pipeline.lightning.log_every_n_steps,
                          callbacks=checkpoint_callback)
    else:
        print("Using CUDA")
        trainer = Trainer(max_epochs=config.pipeline.epochs,
                          devices='auto',   # override config.pipeline.gpus,
                          strategy="ddp",
                          default_root_dir=config.pipeline.save_dir,
                          precision=config.pipeline.precision,
                          logger=loggers,
                          check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                          val_check_interval=1.0,
                          num_sanity_val_steps=config.pipeline.lightning.num_sanity_val_steps,
                          log_every_n_steps=config.pipeline.lightning.log_every_n_steps,
                          callbacks=checkpoint_callback)

    print(f"Trainer created successfully. Starting to fit...")

    trainer.fit(pl_module,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader)
    print(f"Finished")


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
