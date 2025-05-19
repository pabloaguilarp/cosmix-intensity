import argparse
import os
from calendar import day_name
import csv

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d

from configs import get_config
from utils.common.pcd_utils import save_pcd_binary
from utils.datasets.initialization import get_dataset


def parse_numbers(value):
    if value.lower() == "all":
        return "all"
    try:
        return list(map(int, value.split()))
    except ValueError:
        raise argparse.ArgumentTypeError("Must be 'all' or a space-separated list of integers.")


parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/source/synlidar2semantickitti.yaml",
                    type=str,
                    help="Path to config file")
parser.add_argument("--save_dir",
                    default="hist",
                    type=str,
                    help="Folder to save the txt files")
parser.add_argument("--labels",
                    type=parse_numbers,
                    nargs="+",
                    help="Labels to compute histogram")
parser.add_argument("--splits",
                    type=str,
                    nargs="+",
                    choices=["train", "valid", "target"],
                    help="Splits to compute")
parser.add_argument("--save_pcd",
                    type=str,
                    default="",
                    help="Path to folder where individual pcd files are stored. If not set, the pcd files are not stored.")
parser.add_argument("--save_full_scan_pcds",
                    action="store_true",
                    help="If true, pcd files corresponding to complete scans are stored.")
parser.add_argument("--save_labels_pcds",
                    action="store_true",
                    help="If true, pcd files corresponding to individual labels are stored.")
parser.add_argument("--save_hists",
                    action="store_true",
                    help="If true, hist files corresponding to individual labels are stored.")
parser.add_argument("--verbose",
                    action="store_true",
                    help="Verbose mode")
parser.add_argument("--voxel_size",
                    type=float,
                    default=0,
                    help="Voxel grid size in meters")
parser.add_argument("--dbscan_epsilon",
                    type=float,
                    default=0.0,
                    help="DBSCAN epsilon")
parser.add_argument("--dbscan_min_points",
                    type=int,
                    default=0,
                    help="DBSCAN min_points")
parser.add_argument("--cluster_min_points",
                    type=int,
                    default=0,
                    help="Clusters smaller than this parameter are discarded")
parser.add_argument("--limit_scans",
                    type=int,
                    default=0,
                    help="Maximum number of scans to compute. If set to 0 then no limit is applied.")
parser.add_argument("--scans_indexes",
                    type=int,
                    nargs="+",
                    help="Indexes of scans to compute. If not set then default ordered criteria is applied.")
parser.add_argument("--sort_by_content",
                    action="store_true",
                    help="Get a list of the most populated scans for given labels")
parser.add_argument("--compute_distributions",
                    action="store_true",
                    help="Compute light and dark distributions using --threshold and --threshold_method parameters.")
parser.add_argument("--use_two_distributions",
                    action="store_true",
                    help="By default, only one distribution is computed. If this parameter is set, two distributions are computed: one for dark points and one for light points.")
parser.add_argument("--threshold",
                    type=float,
                    default=0.5,
                    help="Threshold to differ between dark and light distributions. This parameter is used only if --compute_distributions is set to True and --use_two_distributions is set to True.")
parser.add_argument("--threshold_method",
                    type=str,
                    choices=["mean", "median"],
                    default="mean",
                    help="Method to differ between dark and light distributions. This parameter is used only if --compute_distributions is set to True and --use_two_distributions is set to True.")
parser.add_argument("--stats_output",
                    type=str,
                    default="",
                    help="Path to the output CSV file for statistics. If not specified, defaults to --save_pcd folder with a filename containing classes and DBSCAN parameters.")

def sort_by_content(dataset,
         labels=None,
         coords_name="coordinates",
         labels_name="labels",):
    num_points_register = []
    for i in tqdm(range(0, len(dataset))):
        scan = dataset.get_data(i)

        labels_num_points = 0
        for label in labels:
            mask = scan[labels_name] == label
            label_num_points = len(scan[coords_name][mask])
            labels_num_points += label_num_points

        num_points_register.append({"index": i, "num_points": labels_num_points})
    num_points_register.sort(key=lambda x: x["num_points"], reverse=True)
    print(f"Most populated scans for labels: {labels}:")
    print(num_points_register)


def loop(dataset,
         args,
         ts_hist=None,
         num_points=0,
         num_label_points=None,
         clusters_count=None,
         clusters_dark_light=None,
         labels=None,
         coords_name="coordinates",
         features_name="features",
         labels_name="labels",):
    if ts_hist is None:
        ts_hist = {}
    if labels is None:
        labels = []
    if num_label_points is None:
        num_label_points = {}
    pcd_counter = 0
    clusters_data = {}

    if args.limit_scans > 0:
        limit_scans = args.limit_scans
    else:
        limit_scans = len(dataset)

    if not args.scans_indexes:
        rng = range(0, limit_scans)
    else:
        rng = args.scans_indexes

    for i in tqdm(rng):
        scan = dataset.get_data(i)
        num_points += len(scan[coords_name])

        if args.save_full_scan_pcds:
            pcd_filename = os.path.join(args.save_pcd, f"scan_{i}.pcd")
            save_pcd_binary(pcd_filename,
                            scan[coords_name],
                            scan[features_name],
                            scan[labels_name])

        for label in labels:
            mask = scan[labels_name] == label
            filtered_scan = {
                coords_name: scan[coords_name][mask],
                features_name: scan[features_name][mask],
                labels_name: scan[labels_name][mask]
            }
            if args.save_pcd != "":
                if len(filtered_scan[coords_name]) > 0:
                    scan_clusters = []
                    if args.dbscan_epsilon > 0.0 and args.dbscan_min_points > 0:
                        # Compute clusters
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(filtered_scan[coords_name])

                        clusters = np.array(pcd.cluster_dbscan(eps=args.dbscan_epsilon, min_points=args.dbscan_min_points, print_progress=False))
                        if clusters.max() > 0:
                            if args.verbose:
                                print(f"scan {i} contains {len(filtered_scan[coords_name])} points and {clusters.max() + 1} clusters")
                            for cluster_id in range(0, clusters.max() + 1):
                                cluster = {
                                    coords_name: filtered_scan[coords_name][clusters == cluster_id],
                                    features_name: filtered_scan[features_name][clusters == cluster_id],
                                    labels_name: np.full(len(filtered_scan[labels_name][clusters == cluster_id]), cluster_id)   # Cluster index
                                }
                                if len(cluster[coords_name]) < args.cluster_min_points:
                                    continue
                                scan_clusters.append(cluster)
                                # Increment the count for this specific label
                                if label not in num_label_points:
                                    num_label_points[label] = 0
                                num_label_points[label] += len(cluster[coords_name])

                                # Compute bounding box
                                center = np.mean(cluster[coords_name], axis=0)
                                distance_to_origin = np.linalg.norm(np.array(center))

                                # Compute intensity
                                avg_intensity = np.mean(cluster[features_name])
                                median_intensity = np.median(cluster[features_name])
                                if args.verbose:
                                    print(f"\tCluster {cluster_id} contains {len(cluster[coords_name])} points. Mean intensity: {avg_intensity}. Median intensity: {median_intensity}. Center: {center}. Distance to origin: {distance_to_origin}")
                                if label not in clusters_data.keys():
                                    clusters_data[label] = []
                                clusters_data[label].append((len(cluster[coords_name]), avg_intensity, median_intensity, center[0], center[1], center[2], distance_to_origin))

                        if args.save_labels_pcds:
                            if len(scan_clusters) == 0:
                                print("No clusters computed. Skipping...")
                            else:
                                pcd_filename = os.path.join(args.save_pcd, f"filtered_scan_{i}_label_{label}_{args.dbscan_epsilon}_{args.dbscan_min_points}.pcd")
                                if args.verbose:
                                    print(f"Saving pcd file: '{pcd_filename}")

                                concat_cluster = {
                                    coords_name: np.concatenate([d[coords_name] for d in scan_clusters]),
                                    features_name: np.concatenate([d[features_name] for d in scan_clusters]),
                                    labels_name: np.concatenate([d[labels_name] for d in scan_clusters]),
                                }
                                save_pcd_binary(
                                    pcd_filename,
                                    concat_cluster[coords_name],
                                    concat_cluster[features_name],
                                    concat_cluster[labels_name]
                                )
                                pcd_counter += 1
                        else:
                            if args.verbose:
                                print(f"Skipping save labels PCD file")

                        if args.compute_distributions:
                            if len(scan_clusters) > 0:
                                if clusters_dark_light is None:
                                    clusters_dark_light = {}

                                if args.use_two_distributions:
                                    if args.verbose:
                                        print(f"Determining light/dark distributions using {args.threshold_method} with threshold {args.threshold}...")
                                    scan_light_clusters_count = 0
                                    scan_dark_clusters_count = 0
                                    for cluster in scan_clusters:
                                        if args.threshold_method == "mean":
                                            thr_intensity = np.mean(cluster[features_name])
                                        elif args.threshold_method == "median":
                                            thr_intensity = np.median(cluster[features_name])
                                        else:
                                            print(f"Threshold method {args.threshold_method} not implemented")
                                            continue
                                        if args.verbose:
                                            print(f"\tThreshold intensity: {thr_intensity}")
                                        if thr_intensity > args.threshold:
                                            scan_light_clusters_count += 1
                                        else:
                                            scan_dark_clusters_count += 1
                                    if label in clusters_dark_light.keys():
                                        clusters_dark_light[label]["dark"] += scan_dark_clusters_count
                                        clusters_dark_light[label]["light"] += scan_light_clusters_count
                                    else:
                                        clusters_dark_light[label] = {}
                                        clusters_dark_light[label]["dark"] = scan_dark_clusters_count
                                        clusters_dark_light[label]["light"] = scan_light_clusters_count
                                else:
                                    if args.verbose:
                                        print(f"Determining distribution...")
                                    scan_clusters_count = 0
                                    for cluster in scan_clusters:
                                        scan_clusters_count += 1
                                    if label in clusters_dark_light.keys():
                                        clusters_dark_light[label]["clusters"] += scan_clusters_count
                                    else:
                                        clusters_dark_light[label] = {}
                                        clusters_dark_light[label]["clusters"] = scan_clusters_count


            if label not in ts_hist.keys():
                ts_hist[label] = filtered_scan[features_name]
            else:
                ts_hist[label] = np.concatenate((ts_hist[label], filtered_scan[features_name]), axis=0)
    print(f"{pcd_counter} pcd files are stored in {args.save_pcd}")
    if clusters_count is None:
        clusters_count = {}
    for label, clusters_val in clusters_data.items():
        clusters_array = np.array(clusters_val)
        if label in clusters_count.keys():
            clusters_count[label] += len(clusters_array)
        else:
            clusters_count[label] = len(clusters_array)
        np.savetxt(os.path.join(args.save_pcd, f"clusters_label_{label}_eps_{args.dbscan_epsilon}_minp_{args.dbscan_min_points}.csv"), clusters_array, delimiter=",", fmt="%.6f", header="cluster_size,mean_intensity,median_intensity,center_x,center_y,center_z,distance_to_origin", comments="")

    return ts_hist, num_points, num_label_points, clusters_count, clusters_dark_light

def process_arg_labels(args_labels, out_classes):
    if len(args_labels) == 1 and args_labels[0] == "all":
        return list(range(0, out_classes))
    else:
        return [d[0] for d in args_labels]

def find_key_by_value(dic, value):
    for key, val in dic.items():
        if val == value:
            return key
    return None  # If value is not found

def check_dataset_path(path):
    if os.path.isdir(os.path.join(path, "sequences")):
        return os.path.join(path, "sequences")
    else:
        return path

def check_target_dataset_path(path):
    if os.path.basename(os.path.normpath(path)) == "sequences":
        return os.path.dirname(path)
    else:
        return path

def main(args):
    config = get_config(args.config_file)
    print(f"Computing contents for dataset {config.dataset.name}")
    print("Loading dataset...")
    source_path = check_dataset_path(config.dataset.dataset_path)
    target_path = check_target_dataset_path(config.dataset.target_path)
    training_dataset, validation_dataset, target_dataset = get_dataset(dataset_name=config.dataset.name,
                                                          dataset_path=source_path,
                                                          target_name=config.dataset.target,
                                                          target_path=target_path,
                                                          voxel_size=config.dataset.voxel_size,
                                                          augment_data=config.dataset.augment_data,
                                                          version='mini',
                                                          sub_num=config.dataset.num_pts,
                                                          num_classes=config.model.out_classes,
                                                          ignore_label=config.dataset.ignore_label,
                                                          mapping_path=config.dataset.mapping_path,
                                                          use_intensity=True,)   # Use same transforms for source and target

    if args.save_pcd != "":
        os.makedirs(args.save_pcd, exist_ok=True)

    labels = process_arg_labels(args.labels, config.model.out_classes)

    if args.sort_by_content:
        print("Sorting by content...")
        sort_by_content(training_dataset, labels=labels)
        exit(0)

    ts_hist = None
    num_points = 0
    num_label_points = {}  # Initialize as a dictionary to track points per label
    clusters_count = None
    clusters_dark_light = None
    if "train" in args.splits:
        print("Processing training split...")
        ts_hist, num_points, num_label_points, clusters_count, clusters_dark_light = loop(training_dataset, args, labels=labels)
    if "valid" in args.splits:
        print("Processing validation split...")
        ts_hist, num_points, num_label_points, clusters_count, clusters_dark_light = loop(validation_dataset, args, num_points=num_points, num_label_points=num_label_points, clusters_count=clusters_count, clusters_dark_light=clusters_dark_light, ts_hist=ts_hist, labels=labels)
    if "target" in args.splits:
        print("Processing validation split...")
        ts_hist, num_points, num_label_points, clusters_count, clusters_dark_light = loop(target_dataset, args, num_points=num_points, clusters_count=clusters_count, clusters_dark_light=clusters_dark_light, ts_hist=ts_hist, labels=labels)

    if ts_hist is None or num_points is None:
        print("No splits specified!")
        exit(0)

    print("SemanticKITTI training + validation splits computed")
    print("********* Point cloud *********")
    print(f"Number of points: {num_points}")

    print("********* Labels *********")
    labels_points = {}
    for label, hist in ts_hist.items():
        labels_points[label] = len(hist)

    for label, count in clusters_count.items():
        label_name = validation_dataset.maps["labels"][find_key_by_value(validation_dataset.maps["learning_map"], label)]
        print(f"Label '{label_name}' ({label}) clustering results:")
        print(f"  Number of label '{label_name}' {label} points: {labels_points[label]} ({(labels_points[label] / num_points * 100):.2f}%)")
        if clusters_dark_light is None:
            print(f"  Number of label '{label_name}' {label} clusters: {count}")
        else:
            if args.use_two_distributions:
                print(f"  Number of label '{label_name}' {label} clusters: {count} - {clusters_dark_light[label]['dark']} dark and {clusters_dark_light[label]['light']} light "
                      f"(light prob. {(clusters_dark_light[label]['light'] / count * 100):.2f}%)")
            else:
                print(f"  Number of label '{label_name}' {label} clusters: {count} - {clusters_dark_light[label]['clusters']} clusters")
        # Get the number of clustered points for this specific label
        label_clustered_points = num_label_points.get(label, 0)
        print(f"  Clustered points of label '{label_name}' {label}: {label_clustered_points} ({(label_clustered_points / labels_points[label] * 100):.2f}%)")


    print("********* Histogram *********")
    # Create a list to store statistics for CSV export
    stats_data = []

    for label, hist in ts_hist.items():
        label_name = validation_dataset.maps["labels"][find_key_by_value(validation_dataset.maps["learning_map"], label)]
        save_file = os.path.join(args.save_dir, f"hist_{label}_{label_name}.png")

        # Calculate and display mean and standard deviation
        mean = np.mean(hist)
        std = np.std(hist)
        print(f"Label '{label_name}' {label} distribution statistics:")
        print(f"  Mean: {mean:.4f}, Standard Deviation: {std:.4f}")

        # Initialize stats dictionary for this label
        label_stats = {
            'label_id': label,
            'label_name': label_name,
            'mean': mean,
            'std': std
        }

        # If using two distributions, calculate stats for each
        if args.use_two_distributions and args.compute_distributions:
            # Separate dark and light points based on threshold
            # Apply the threshold directly to intensity values
            mask_light = hist > args.threshold

            dark_hist = hist[~mask_light]
            light_hist = hist[mask_light]

            if len(dark_hist) > 0:
                dark_mean = np.mean(dark_hist)
                dark_std = np.std(dark_hist)
                print(f"  Dark distribution - Mean: {dark_mean:.4f}, Standard Deviation: {dark_std:.4f}")
                label_stats['dark_mean'] = dark_mean
                label_stats['dark_std'] = dark_std
            else:
                label_stats['dark_mean'] = 'N/A'
                label_stats['dark_std'] = 'N/A'

            if len(light_hist) > 0:
                light_mean = np.mean(light_hist)
                light_std = np.std(light_hist)
                print(f"  Light distribution - Mean: {light_mean:.4f}, Standard Deviation: {light_std:.4f}")
                label_stats['light_mean'] = light_mean
                label_stats['light_std'] = light_std
            else:
                label_stats['light_mean'] = 'N/A'
                label_stats['light_std'] = 'N/A'

        # Add this label's stats to the overall data
        stats_data.append(label_stats)

        if not args.save_hists:
            if args.verbose:
                print("Histograms are not saved")
        else:
            if args.verbose:
                print(f"Saving histogram to {save_file}")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            plt.hist(hist, bins=100)
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            plt.title(f"{label_name}_{label}")
            plt.savefig(save_file)  # Save as PNG
            plt.close()

    # Save statistics to CSV file
    if stats_data:
        # Determine the output file path
        if args.stats_output:
            stats_file = args.stats_output
        else:
            # Default filename based on save_pcd folder and parameters
            if args.save_pcd:
                # Create a string with the labels used
                labels_str = "_".join([str(label) for label in labels])
                stats_file = os.path.join(args.save_pcd, f"stats_labels_{labels_str}_eps_{args.dbscan_epsilon}_minp_{args.dbscan_min_points}.csv")
            else:
                # If save_pcd is not set, use the current directory
                labels_str = "_".join([str(label) for label in labels])
                stats_file = f"stats_labels_{labels_str}_eps_{args.dbscan_epsilon}_minp_{args.dbscan_min_points}.csv"

        # Ensure directory exists
        os.makedirs(os.path.dirname(stats_file) if os.path.dirname(stats_file) else '.', exist_ok=True)

        # Determine the fieldnames based on whether we have dark/light distributions
        fieldnames = ['label_id', 'label_name', 'mean', 'std']
        if args.use_two_distributions and args.compute_distributions:
            fieldnames.extend(['dark_mean', 'dark_std', 'light_mean', 'light_std'])

        print(f"Saving statistics to {stats_file}")
        with open(stats_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for stats in stats_data:
                # Filter the dictionary to include only the fields in fieldnames
                filtered_stats = {k: v for k, v in stats.items() if k in fieldnames}
                writer.writerow(filtered_stats)
        print(f"Statistics saved to {stats_file}")

    print("*" * 30)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
