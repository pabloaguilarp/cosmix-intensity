# Histogram Plotting Tool

## Overview
`plot_hist.py` is a command-line tool for generating histograms from CSV data files. It provides various customization options and can compute statistics on the data. This tool is particularly useful for analyzing distributions of data points in the CoSMix project for domain adaptation in 3D LiDAR segmentation.

## Features
- Generate histograms from CSV data
- Filter data based on column values
- Compute statistics for one or two distributions
- Customize histogram appearance (bins, labels, title)
- Save histogram images to file

## Dependencies
- pandas
- matplotlib
- numpy
- argparse

## Usage
```bash
python plot_hist.py --input_file <path_to_csv> --column <column_name> [options]
```

### Required Arguments
- `--input_file`: Path to the input CSV file
- `--column`: Column name to plot histogram

### Optional Arguments
- `--bins`: Number of histogram bins (default: 100)
- `--save_path`: Path to output histogram image file
- `--title`: Chart title (default: filename - column)
- `--xlabel`: Chart X-axis label (default: "Intensity")
- `--ylabel`: Chart Y-axis label (default: "Frequency")
- `--filter`: Filter by column. Format: '<column_name> gt|st|eq|neq <value>'. Multiple filters can be applied by providing multiple filter strings.
- `--compute_statistics`: Calculate statistics
- `--use_two_distributions`: Use two distributions (dark/light points)
- `--threshold`: Threshold for light/dark separation (default: 0.5)

## Examples

### Basic Usage
Generate a histogram for the "mean" column in data.csv:
```bash
python plot_hist.py --input_file data.csv --column mean
```

### Save Histogram to File
Generate a histogram and save it to an image file:
```bash
python plot_hist.py --input_file data.csv --column mean --save_path histogram.png
```

### Customize Appearance
Customize the histogram appearance:
```bash
python plot_hist.py --input_file data.csv --column mean --bins 50 --title "Distribution of Mean Values" --xlabel "Mean Value" --ylabel "Count"
```

### Filter Data
Filter data before generating the histogram:
```bash
python plot_hist.py --input_file data.csv --column mean --filter "cluster_size gt 10"
```

### Compute Statistics
Calculate statistics for the data:
```bash
python plot_hist.py --input_file data.csv --column mean --compute_statistics
```

### Two Distributions
Calculate statistics for two distributions (light and dark):
```bash
python plot_hist.py --input_file data.csv --column mean --compute_statistics --use_two_distributions --threshold 0.6
```

## Filters in Detail

The `--filter` option allows you to filter data based on column values before generating the histogram. This is useful for focusing on specific subsets of your data.

### Filter Format
Each filter follows the format: `<column_name> <operator> <value>`

### Available Filter Operators
- `gt`: Greater than or equal to (>=)
- `st`: Smaller than or equal to (<=)
- `eq`: Equal to (==)
- `neq`: Not equal to (!=)

### Examples of Filter Usage

#### Greater Than Filter
Filter data where the "cluster_size" column has values greater than or equal to 10:
```bash
python plot_hist.py --input_file data.csv --column mean --filter "cluster_size gt 10"
```

#### Smaller Than Filter
Filter data where the "mean" column has values smaller than or equal to 0.5:
```bash
python plot_hist.py --input_file data.csv --column mean --filter "mean st 0.5"
```

#### Equal To Filter
Filter data where the "class_id" column has values equal to 1:
```bash
python plot_hist.py --input_file data.csv --column mean --filter "class_id eq 1"
```

#### Not Equal To Filter
Filter data where the "class_id" column has values not equal to 0:
```bash
python plot_hist.py --input_file data.csv --column mean --filter "class_id neq 0"
```

### Using Multiple Filters
You can apply multiple filters by providing multiple filter strings. All filters are applied sequentially (AND logic):

```bash
python plot_hist.py --input_file data.csv --column mean --filter "cluster_size gt 10" "mean st 0.8" "class_id neq 0"
```

This example filters data where:
1. "cluster_size" is greater than or equal to 10, AND
2. "mean" is smaller than or equal to 0.8, AND
3. "class_id" is not equal to 0

### Notes on Filtering
- All filter values are converted to floating-point numbers, so they work best with numeric columns
- If a column specified in a filter doesn't exist in the data, the filter is ignored
- If a filter format is incorrect (doesn't have exactly 3 parts), it is ignored
- Unsupported operators will be reported but ignored
