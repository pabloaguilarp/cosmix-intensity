import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import MinkowskiEngine as ME

from utils.collation import CollateFN
from utils.common.check_ddp import is_cuda_available
from utils.datasets.initialization import get_dataset
from configs import get_config
from utils.utils import build_model
from utils.pipelines import PLTTester


parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/source/synlidar_semantickitti.yaml",
                    type=str,
                    help="Path to config file")
parser.add_argument("--eval_source",
                    action='store_true',
                    default=False)
parser.add_argument("--eval_target",
                    action='store_true',
                    default=False)
parser.add_argument("--resume_path",
                    type=str,
                    default=None)
parser.add_argument("--is_student",
                    default=False,
                    action='store_true')
parser.add_argument("--save_predictions",
                    default=False,
                    action='store_true')


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
    model.load_state_dict(ckpt, strict=False)
    return model


def load_student_model(checkpoint_path, model):
    # reloads model
    def clean_state_dict(state):
        # clean state dict from names of PL
        for k in list(ckpt.keys()):
            if "model" in k:
                ckpt[k.replace("student_model.", "")] = ckpt[k]
            del ckpt[k]
        return state

    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
    ckpt = clean_state_dict(ckpt)
    model.load_state_dict(ckpt, strict=True)
    return model


def test(config, resume_checkpoint):
    collate = CollateFN()
    def get_dataloader(dataset, shuffle=False, pin_memory=True):
        return DataLoader(dataset,
                          batch_size=config.pipeline.dataloader.batch_size*4,
                          collate_fn=collate,
                          shuffle=shuffle,
                          num_workers=config.pipeline.dataloader.num_workers,
                          pin_memory=pin_memory)
    try:
        mapping_path = config.dataset.mapping_path
    except AttributeError('--> Setting default class mapping path!'):
        mapping_path = None

    print(f"Source: {config.dataset.name} at {config.dataset.dataset_path}")
    print(f"Target: {config.dataset.target} at {config.dataset.target_path}")

    _, validation_dataset, target_dataset = get_dataset(dataset_name=config.dataset.name,
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
                                                        feat_keys=config.model.features,)

    validation_dataloader = get_dataloader(validation_dataset, shuffle=False)
    target_dataloader = get_dataloader(target_dataset, shuffle=False)

    validation_dataloader = [validation_dataloader]
    target_dataloader = [target_dataloader]

    model = build_model(config.model)
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    if not args.is_student:
        model = load_model(resume_checkpoint, model)
        print(f'--> LOADED MODEL FROM {resume_checkpoint}')
    else:
        model = load_student_model(resume_checkpoint, model)
        print(f'--> LOADED STUDENT MODEL FROM {resume_checkpoint}')

    dataset = validation_dataset if args.eval_source else target_dataset

    main_dir, _ = os.path.split(resume_checkpoint)

    save_dir = os.path.join(main_dir, 'evaluation')
    if not args.is_student:
        if args.eval_source:
            save_dir = os.path.join(save_dir, 'source')
        elif args.eval_target:
            save_dir = os.path.join(save_dir, 'target')
        else:
            print('Not evaluating!')
    save_preds_dir = os.path.join(main_dir, 'predictions')

    plt_model = PLTTester(model,
                          criterion=config.pipeline.loss,
                          dataset=dataset,
                          num_classes=config.model.out_classes,
                          checkpoint_path=resume_checkpoint,
                          save_predictions=args.save_predictions,
                          save_folder=save_preds_dir,
                          coords_name="coordinates",
                          features_name="features",
                          labels_name="labels")

    wandb_logger = WandbLogger(project=config.pipeline.wandb.project_name,
                               entity=config.pipeline.wandb.entity_name,
                               name=save_dir,
                               offline=True)

    os.makedirs(save_dir, exist_ok=True)
    loggers = [wandb_logger]

    if not is_cuda_available():
        tester = Trainer(max_epochs=config.pipeline.epochs,
                         accelerator='cpu',
                         logger=loggers,
                         default_root_dir=save_dir,
                         val_check_interval=1.0,
                         num_sanity_val_steps=0)
    else:
        tester = Trainer(max_epochs=config.pipeline.epochs,
                         devices=[0],
                         logger=loggers,
                         default_root_dir=save_dir,
                         val_check_interval=1.0,
                         num_sanity_val_steps=0)

    if args.eval_source:
        tester.test(plt_model, dataloaders=validation_dataloader, verbose=False)
    elif args.eval_target:
        tester.test(plt_model, dataloaders=target_dataloader, verbose=False)
    else:
        print('Not evaluating!')


def multiple_test(config, path, args):
    list_checkpoint = os.listdir(os.path.join(path, 'checkpoints'))
    list_checkpoint = [c for c in list_checkpoint if c.endswith('.ckpt')]
    for checkpoint in list_checkpoint:
        evaluation_path = os.path.join(path, 'checkpoints', 'evaluation')
        if not args.is_student:
            if args.eval_source:
                evaluation_path = os.path.join(evaluation_path, 'source')
            elif args.eval_target:
                evaluation_path = os.path.join(evaluation_path, 'target')
            else:
                print('Not evaluating!')
        test_file_path = os.path.join(evaluation_path, 'results', checkpoint[:-5]+'_test.csv')
        if not os.path.isfile(test_file_path):
            print(f'############### EVALUATING {checkpoint} ####################')
            test(config, os.path.join(path, 'checkpoints', checkpoint))
        else:
            print(f'############### EVALUATION {checkpoint} SKIPPED - ALREADY PRESENT ####################')


if __name__ == '__main__':
    args = parser.parse_args()

    config = get_config(args.config_file)

    # fix random seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True

    multiple_test(config, args.resume_path, args)
