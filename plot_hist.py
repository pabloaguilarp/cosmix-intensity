import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_file",
                    required=True,
                    type=str,
                    help="Path to input CSV file")
parser.add_argument("--column",
                    required=True,
                    type=str,
                    help="Column name to plot histogram")
parser.add_argument("--bins",
                    type=int,
                    default=100,
                    help="Number of histogram bins")
parser.add_argument("--save_path",
                    type=str,
                    default=None,
                    help="Path to output histogram image file")
parser.add_argument("--title",
                    type=str,
                    default=None,
                    help="Chart title")
parser.add_argument("--xlabel",
                    type=str,
                    default="Intensity",
                    help="Chart X-axis label")
parser.add_argument("--ylabel",
                    type=str,
                    default="Frequency",
                    help="Chart Y-axis label")
parser.add_argument("--filter",
                    type=str,
                    nargs="+",
                    help="Filter by column. Format: '<column_name> gt|st|eq|neq <value>'")
parser.add_argument("--compute_statistics",
                    action="store_true",
                    help="Calculate statistics")
parser.add_argument("--use_two_distributions",
                    action="store_true",
                    help="By default, only one distribution is computed. If this parameter is set, two distributions are computed: one for dark points and one for light points.")
parser.add_argument("--threshold",
                    type=float,
                    default=0.5,
                    help="Threshold for light/dark")

def process_filter(filters, df):
    for filter_str in filters:
        parts = filter_str.split(" ")
        if len(parts) != 3:
            continue
        if parts[0] not in df.columns:
            continue
        if parts[1] == "gt":
            print(f"Filtering column '{parts[0]}' for values greater than {parts[2]}")
            df = df[df[parts[0]] >= float(parts[2])]
        elif parts[1] == "st":
            print(f"Filtering column '{parts[0]}' for values smaller than {parts[2]}")
            df = df[df[parts[0]] <= float(parts[2])]
        elif parts[1] == "eq":
            print(f"Filtering column '{parts[0]}' for values equal to {parts[2]}")
            df = df[df[parts[0]] == float(parts[2])]
        elif parts[1] == "neq":
            print(f"Filtering column '{parts[0]}' for values distinct from {parts[2]}")
            df = df[df[parts[0]] != float(parts[2])]
        else:
            print(f"Operator '{parts[1]}' not supported")
    return df


def compute(ld_data, column, weights):
    # Values and weights
    ld_values = ld_data[column].to_numpy()
    ld_weights = ld_data[weights].to_numpy()

    # Weighted average
    ld_weighted_avg = np.average(ld_values, weights=ld_weights)

    # Weighted standard deviation
    ld_weighted_var = np.average((ld_values - ld_weighted_avg) ** 2, weights=ld_weights)
    ld_weighted_std = np.sqrt(ld_weighted_var)
    return ld_weighted_avg, ld_weighted_std

def compute_statistics_two_dists(data, threshold, column: str="mean", weights: str="cluster_size"):
    light_data = data[data[column] > threshold]
    dark_data = data[data[column] <= threshold]

    light_weighted_avg, light_weighted_std = compute(light_data, column, weights)
    dark_weighted_avg, dark_weighted_std = compute(dark_data, column, weights)

    print("**************** General *****************")
    print(f"Light clusters: {len(light_data)} ({(len(light_data) / len(data) * 100):.2f}%)")
    print(f"Dark clusters: {len(dark_data)} ({(len(dark_data) / len(data) * 100):.2f}%)")
    print("*********** Light distribution ***********")
    print(f"μ={light_weighted_avg:.2f}")
    print(f"σ={light_weighted_std:.2f}")
    print("*********** Dark distribution ************")
    print(f"μ={dark_weighted_avg:.2f}")
    print(f"σ={dark_weighted_std:.2f}")
    print("******************************************")


def compute_statistics_one_dist(data, column: str="mean", weights: str="cluster_size"):
    weighted_avg, weighted_std = compute(data, column, weights)

    print("**************** General *****************")
    print(f"Total data points: {len(data)} (100.00%)")
    print("************ Distribution ****************")
    print(f"μ={weighted_avg:.2f}")
    print(f"σ={weighted_std:.2f}")
    print("******************************************")


def main(args):
    file_path = args.input_file
    df = pd.read_csv(file_path)

    df = process_filter(args.filter, df)

    if args.compute_statistics:
        if args.use_two_distributions:
            compute_statistics_two_dists(df, args.threshold, args.column)
        else:
            compute_statistics_one_dist(df, args.column)

    column_name = args.column
    data = df[column_name]

    plt.hist(data, bins=args.bins)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(args.title if args.title else f"{os.path.basename(args.input_file)} - {args.column}")
    if args.save_path:
        plt.savefig(args.save_path)  # Save as PNG
    plt.show()
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)