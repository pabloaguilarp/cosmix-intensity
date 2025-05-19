# CoSMix Intensity Experiments Guide

This document provides a comprehensive guide to running experiments with the CoSMix framework for domain adaptation in 3D LiDAR segmentation.

## Overview

The CoSMix framework provides tools for training, adapting, and evaluating models for domain adaptation in 3D LiDAR segmentation. The framework supports both Unsupervised Domain Adaptation (UDA) and Semi-Supervised Domain Adaptation (SSDA) approaches.

The experimental workflow typically consists of the following steps:

1. **Training a source model**: Train a model on the source domain (e.g., SynLiDAR)
2. **Adapting the model**: Adapt the trained model to the target domain (e.g., SemanticKITTI) using either UDA or SSDA
3. **Fine-tuning the model**: Optionally fine-tune the adapted model on the target domain
4. **Evaluating the model**: Evaluate the model's performance on the target domain

## Main Python Scripts

### train_source_v3.py

This script is used to train a model on the source domain (e.g., SynLiDAR).

```bash
python train_source_v3.py \
  --config_file <path_to_config_file> \
  --use_intensity <intensity_mode> \
  [--custom_intensity_path <path_to_custom_intensity>] \
  [--use_intensity_postprocess <true|false>] \
  [--postprocess_labels <comma_separated_labels>] \
  [--postprocess_params_path <path_to_params_file>]
```

### adapt_cosmix_uda.py

This script is used to adapt a trained model to the target domain using Unsupervised Domain Adaptation (UDA).

```bash
python adapt_cosmix_uda.py \
  --config_file <path_to_config_file> \
  --checkpoint_path <path_to_checkpoint> \
  --use_intensity <intensity_mode> \
  [--custom_intensity_path <path_to_custom_intensity>] \
  [--use_intensity_postprocess <true|false>] \
  [--postprocess_labels <comma_separated_labels>] \
  [--postprocess_params_path <path_to_params_file>]
```

### adapt_cosmix_ssda.py

This script is used to adapt a trained model to the target domain using Semi-Supervised Domain Adaptation (SSDA).

```bash
python adapt_cosmix_ssda.py \
  --config_file <path_to_config_file> \
  --checkpoint_path <path_to_checkpoint> \
  --use_intensity <intensity_mode> \
  [--custom_intensity_path <path_to_custom_intensity>] \
  [--use_intensity_postprocess <true|false>] \
  [--postprocess_labels <comma_separated_labels>] \
  [--postprocess_params_path <path_to_params_file>] \
  [--num_frames <number_of_frames>]
```

### finetune_ssda.py

This script is used to fine-tune an adapted model on the target domain using Semi-Supervised Domain Adaptation (SSDA).

```bash
python finetune_ssda.py \
  --config_file <path_to_config_file> \
  --checkpoint_path <path_to_checkpoint> \
  --use_intensity <intensity_mode> \
  [--custom_intensity_path <path_to_custom_intensity>] \
  [--use_intensity_postprocess <true|false>] \
  [--postprocess_labels <comma_separated_labels>] \
  [--postprocess_params_path <path_to_params_file>]
```

## Command-Line Arguments

### Common Arguments

- `--config_file`: Path to the configuration file
- `--use_intensity`: Intensity mode (none, default, custom)
- `--custom_intensity_path`: Path to custom intensity predictions
- `--use_intensity_postprocess`: Whether to apply intensity postprocessing (true, false)
- `--postprocess_labels`: Comma-separated list of labels to apply intensity postprocessing to (e.g., '18,19,20')
- `--postprocess_params_path`: Path to YAML file with postprocessing parameters

### Adaptation-Specific Arguments

- `--checkpoint_path`: Path to the checkpoint file for both teacher and student models
- `--num_frames`: Number of target frames to include in the adaptation process (SSDA only)

## Intensity Postprocessing

The framework supports intensity postprocessing to improve the realism of synthetic data. This is particularly useful for specific classes like traffic signs.

### Intensity Modes

- `none`: Do not use intensity information
- `default`: Use default intensity information from the dataset
- `custom`: Use custom intensity predictions from a specified path

### Postprocessing Parameters

The intensity postprocessing parameters are stored in a YAML file (default: `utils/datasets/_resources/intensity_postprocess.yaml`). The file contains parameters for each label, including:

- `label`: The label ID
- `dbscan_epsilon`: DBSCAN epsilon parameter for clustering
- `dbscan_min_points`: DBSCAN minimum points parameter for clustering
- `use_two_distributions`: Whether to use two distributions (light/dark) or just one
- `mean_dark`, `std_dev_dark`, `mean_light`, `std_dev_light`, `light_prob`: Parameters for two distributions
- `mean`, `std_dev`: Parameters for single distribution

### UDA Workflow

1. Train a source model:
   ```bash
   python train_source_v3.py \
     --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
     --use_intensity default \
     --use_intensity_postprocess false
   ```

2. Adapt the model to the target domain using UDA:
   ```bash
   python adapt_cosmix_uda.py \
     --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
     --checkpoint_path checkpoints/source_model.pth \
     --use_intensity default \
     --use_intensity_postprocess false
   ```

3. Evaluate the adapted model:
   ```bash
   python evaluate_model.py \
     --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
     --checkpoint_path checkpoints/adapted_model.pth \
     --use_intensity default
   ```

### SSDA Workflow

1. Train a source model:
   ```bash
   python train_source_v3.py \
     --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
     --use_intensity default \
     --use_intensity_postprocess false
   ```

2. Fine-tune the adapted model:
   ```bash
   python finetune_ssda.py \
     --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
     --checkpoint_path checkpoints/source_model.pth \
     --use_intensity default \
     --use_intensity_postprocess false
   ```

3. Adapt the model to the target domain using SSDA:
   ```bash
   python adapt_cosmix_ssda.py \
     --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
     --checkpoint_path checkpoints/finetuned_model.pth \
     --use_intensity default \
     --use_intensity_postprocess false \
     --num_frames 2
   ```
   
4. Evaluate the adapted model:
   ```bash
   python eval.py \
     --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
     --checkpoint_path checkpoints/ssda_model.pth \
     --use_intensity default
   ```

### Using Intensity Postprocessing

To use intensity postprocessing for specific labels:

1. Train a source model with intensity postprocessing:
   ```bash
   python train_source_v3.py \
     --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
     --use_intensity default \
     --use_intensity_postprocess true
   ```

2. Adapt the model with intensity postprocessing:
   ```bash
   python adapt_cosmix_uda.py \
     --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
     --checkpoint_path checkpoints/source_model.pth \
     --use_intensity default \
     --use_intensity_postprocess true
   ```

3. Evaluate the adapted model:
   ```bash
   python evaluate_model.py \
     --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
     --checkpoint_path checkpoints/adapted_model.pth \
     --use_intensity default
   ```

## Advanced Usage

### Custom Intensity Predictions

To use custom intensity predictions:

```bash
python train_source_v3.py \
  --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
  --use_intensity custom \
  --custom_intensity_path /path/to/custom/intensity/predictions
```

### Custom Postprocessing Parameters

To use custom postprocessing parameters:

```bash
python train_source_v3.py \
  --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
  --use_intensity default \
  --use_intensity_postprocess true \
  --postprocess_params_path /path/to/custom/params.yaml
```

### Per-label Postprocessing Parameters

To specify which labels to apply postprocessing to:

```bash
python train_source_v3.py \
  --config_file configs/source/exp/synlidar2semantickitti_minkunet34.yaml \
  --use_intensity default \
  --use_intensity_postprocess true \
  --postprocess_labels "18,19,20"
```