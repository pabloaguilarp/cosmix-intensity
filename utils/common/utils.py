import os
import time

from torch import use_deterministic_algorithms
from torch.utils.data import DataLoader
import torch

from configs.config import Config

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import MinkowskiEngine as ME

from utils.collation import CollateFN, CollateMerged
from utils.datasets.initialization import get_dataset, get_concat_dataset
from utils.utils import build_model


def print_dataset_info(dataset, string: str):
    print(f"*** {string} dataset ***")
    print(f"{string.lower()}_dataset: {type(dataset)}")
    print(f"\t{len(dataset)} total samples")
    if hasattr(dataset, "transform"):
        print(f"transforms: {len(dataset.transform.transforms)}")
    else:
        print(f"Dataset has no transforms")
    for key in dataset[0].keys():
        if hasattr(dataset[0][key], "shape"):
            print(f"{key}: {type(dataset[0][key])}, shape: {dataset[0][key].shape}")
        else:
            print(f"{key}: {type(dataset[0][key])}")

def setup_loggers(config: Config, include_run_name=True):
    """
    This function sets up the Wandb logger and the checkpoint callback for trainer.
    The checkpoint path is '${save_dir}/model/<run_name>/checkpoints/model_v1.ckpt'

    @param config: config instance loaded from yaml file
    @param include_run_name: if true, a subfolder named 'run_name' is added
    """
    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    if config.pipeline.wandb.run_name is not None:
        run_name = run_time + '_' + config.pipeline.wandb.run_name
    else:
        run_name = run_time
    run_name = run_name + "_" + config.model.name
    if include_run_name:
        save_dir = os.path.join(config.pipeline.save_dir, str(config.model.name).lower(), run_name)
    else:
        save_dir = os.path.join(config.pipeline.save_dir, str(config.model.name).lower())

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
    print(f"wandb logger created successfully")
    return loggers, checkpoint_callback

def check_intensity_params(args):
    print(f"Using intensity: '{args.use_intensity}'")
    if args.use_intensity == "custom" and args.custom_intensity_path is None:
        print("Intensity mode is set to 'custom' but no custom intensity path is provided. "
              "Please set --custom_intensity_path parameter to use custom intensity predictions.")
        print("\tFalling back to default intensity predictions.")
        args.use_intensity = "default"
    if args.use_intensity == "default" or args.use_intensity == "none":
        args.custom_intensity_path = None

def set_save_dir(config, args, num_frames: int=None, postprocess_labels:list =None):
    use_deterministic_paths = True

    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    if config.pipeline.wandb.run_name is not None:
        run_name_prefix = run_time + '_' + config.pipeline.wandb.run_name
    else:
        run_name_prefix = run_time

    suffix = args.use_intensity
    if args.use_intensity_postprocess == "true":
        suffix = suffix + "_use_pp"
    if num_frames is not None:
        suffix = suffix + "_" + str(num_frames)
    if postprocess_labels is not None:
        labels_str = "labels_" + "_".join(map(str, postprocess_labels))
        suffix = suffix + "_" + labels_str

    run_name = run_name_prefix + "_" + suffix
    if use_deterministic_paths:
        return os.path.join(config.pipeline.save_dir, str(config.model.name).lower(), suffix), run_name
    else:
        return os.path.join(config.pipeline.save_dir, str(config.model.name).lower(), run_name), run_name

def load_data_concat(args, config, postprocess_labels=None):
    def get_dataloader(dataset, batch_size, shuffle=False, pin_memory=True, collation=None):
        if collation is None:
            collation = CollateFN()

        return DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collation,
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

    print(f"Loading source dataset '{config.source_dataset.name}' from path '{config.source_dataset.dataset_path}'")
    print(f"Loading target dataset '{config.target_dataset.name}' from path '{config.target_dataset.dataset_path}'")

    source_training_dataset, source_validation_dataset, source_testing_dataset = get_dataset(dataset_name=config.source_dataset.name,
                                                                        dataset_path=config.source_dataset.dataset_path,
                                                                        voxel_size=config.source_dataset.voxel_size,
                                                                        augment_data=config.source_dataset.augment_data,
                                                                        version=config.source_dataset.version,
                                                                        sub_num=config.source_dataset.num_pts,
                                                                        num_classes=config.model.out_classes,
                                                                        ignore_label=config.source_dataset.ignore_label,
                                                                        mapping_path=source_mapping_path,
                                                                        target_name=config.target_dataset.name,
                                                                        target_path=os.path.dirname(
                                                                            config.target_dataset.dataset_path),
                                                                        use_intensity=args.use_intensity != "none",
                                                                        use_traffic_sign_postprocessing=args.use_intensity_postprocess == "true",
                                                                        postprocess_labels=postprocess_labels,
                                                                        postprocess_params_path=getattr(args, 'postprocess_params_path', '_resources/intensity_postprocess.yaml'),
                                                                        intensity_path=args.custom_intensity_path,
                                                                        weights_path=config.source_dataset.weights_path,
                                                                        feat_keys=config.model.features)  # Use same transforms for source and target

    target_training_dataset, target_validation_dataset, target_testing_dataset = get_dataset(dataset_name=config.target_dataset.name,
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
                                                                        postprocess_params_path=getattr(args, 'postprocess_params_path', '_resources/intensity_postprocess.yaml'),
                                                                        intensity_path=args.custom_intensity_path,
                                                                        feat_keys=config.model.features)

    training_dataset = get_concat_dataset(source_dataset=source_training_dataset,
                                          target_dataset=target_training_dataset,
                                          augment_data=config.masked_dataset.augment_data,
                                          augment_mask_data=config.masked_dataset.augment_mask_data,
                                          remove_overlap=config.masked_dataset.remove_overlap,
                                          feat_keys=config.model.features,
                                          coords_name="coordinates",
                                          features_name="features",
                                          labels_name="labels")

    training_dataloader = get_dataloader(training_dataset,
                                         batch_size=config.pipeline.dataloader.train_batch_size,
                                         shuffle=True,
                                         collation=CollateMerged())

    source_validation_dataloader = get_dataloader(source_validation_dataset,
                                                  batch_size=config.pipeline.dataloader.train_batch_size,
                                                      shuffle=False,
                                                      collation=CollateFN())

    target_validation_dataloader = get_dataloader(target_validation_dataset,
                                                  batch_size=config.pipeline.dataloader.train_batch_size,
                                                  shuffle=False,
                                                  collation=CollateFN())

    validation_dataloaders = [source_validation_dataloader, target_validation_dataloader]

    return training_dataset, source_validation_dataset, target_validation_dataset, training_dataloader, validation_dataloaders, target_training_dataset


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

def load_net_model(config_model: Config, checkpoint_path, mode: str):
    model = build_model(config_model)
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    print(f"{mode.capitalize()} model created successfully. {type(model)}")

    if checkpoint_path is not None:
        model = load_model(checkpoint_path, model)
        print(f'--> Loaded {mode.lower()} checkpoint {checkpoint_path}')
    else:
        print(f'--> Using pristine {mode.lower()} model!')
    return model
