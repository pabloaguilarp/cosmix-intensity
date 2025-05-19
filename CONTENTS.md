# Contents Analysis Tool

## Overview
This script is designed to analyze and visualize point cloud data, particularly focusing on intensity distributions and cluster analysis across different labels in LiDAR datasets.

## Features
- Analysis of point cloud data across different dataset splits (training, validation, target)
- Histogram generation for intensity distributions
- Cluster analysis using DBSCAN
- Support for both single and dual (light/dark) distribution analysis
- PCD file generation and storage
- Statistical analysis of point distributions

## Usage

### Basic Command
```shell script
python contents.py [arguments]
```


### Required Arguments
- `--labels`: Labels to analyze (space-separated integers or "all")
- `--splits`: Dataset splits to process ("train", "valid", "target")

### Optional Arguments

#### Configuration
- `--config_file`: Path to configuration file (default: "configs/source/synlidar2semantickitti.yaml")
- `--save_dir`: Directory to save output files (default: "hist")
- `--verbose`: Enable verbose output

#### PCD File Options
- `--save_pcd`: Directory to store PCD files
- `--save_full_scan_pcds`: Save complete scan PCDs
- `--save_labels_pcds`: Save PCDs for individual labels
- `--save_hists`: Save histogram images

#### Clustering Parameters
- `--voxel_size`: Voxel grid size in meters (default: 0)
- `--dbscan_epsilon`: DBSCAN epsilon parameter (default: 0.0)
- `--dbscan_min_points`: Minimum points for DBSCAN clusters (default: 0)
- `--cluster_min_points`: Minimum cluster size threshold (default: 0)

#### Analysis Control
- `--limit_scans`: Maximum number of scans to process (0 for no limit)
- `--scans_indexes`: Specific scan indices to process
- `--sort_by_content`: List most populated scans for given labels
- `--compute_distributions`: Enable distribution analysis
- `--use_two_distributions`: By default, a single distribution is computed. If enabled, two distributions are computed, split by a threshold.
- `--threshold`: Threshold for light/dark classification (default: 0.5)
- `--threshold_method`: Method for threshold calculation ("mean" or "median")

### Example Commands

1. Basic histogram generation:
```shell script
python contents.py --labels all --splits train valid --save_hists
```


2. Cluster analysis with DBSCAN:
```shell script
python contents.py --labels 0 1 2 --splits train --save_pcd ./output_pcds --dbscan_epsilon 0.5 --dbscan_min_points 10
```


3. Distribution analysis:
```shell script
python contents.py --labels all --splits train --compute_distributions --threshold 0.5 --threshold_method mean
```


## Output

The script generates several types of output:

1. **Histograms**: PNG files showing intensity distributions per label
2. **PCD Files**: Point cloud data files for full scans or individual labels
3. **CSV Files**: Cluster statistics and analysis results
4. **Console Output**: Statistical information including:
   - Total point counts
   - Label-wise point distributions
   - Cluster counts
   - Light/dark distribution statistics (if enabled)

## Implementation Details

### Main Components

1. **Data Loading**: Uses configured dataset loaders for different splits
2. **Processing Loop**: Iterates through scans and labels
3. **Cluster Analysis**: DBSCAN-based clustering with customizable parameters
4. **Distribution Analysis**: Single or dual-distribution analysis of intensity values
5. **Visualization**: Histogram generation and statistical reporting

### Performance Considerations

- Uses TQDM for progress tracking
- Supports scan limiting for testing
- Includes memory-efficient processing of large datasets

## Dependencies

- NumPy
- Matplotlib
- Open3D
- TQDM
- Custom utilities from project