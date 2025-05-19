import open3d as o3d
import numpy as np
from scipy.stats import truncnorm


def sample_truncated_normal(mean, std_dev, size):
    a, b = (0 - mean) / std_dev, (1 - mean) / std_dev  # Compute bounds in std space
    return truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=size)


def post_process_intensity_label(data_dict, label, dbscan_epsilon, dbscan_min_points, 
                                use_two_distributions=True, mean=None, std_dev=None,
                                mean_dark=None, std_dev_dark=None, mean_light=None, 
                                std_dev_light=None, light_prob=None):
    """
    @param data_dict:    {"points": points,       # (N, 3)
                          "colors": colors,     # (N, 1)
                          "labels": labels}      # (N, 1)
    @param label: label to sample
    @param dbscan_epsilon: DBSCAN epsilon
    @param dbscan_min_points: DBSCAN min_points
    @param use_two_distributions: whether to use two distributions (light/dark) or just one
    @param mean: mean for single distribution
    @param std_dev: standard deviation for single distribution
    @param mean_dark: dark mean (for two distributions)
    @param std_dev_dark: dark standard deviation (for two distributions)
    @param mean_light: light mean (for two distributions)
    @param std_dev_light: light standard deviation (for two distributions)
    @param light_prob: probability of sampling a light distribution (for two distributions)
    @return: updated data_dict
    """
    mask = data_dict["labels"] == label
    filtered_scan = {
        "points": data_dict["points"][mask],
        "colors": data_dict["colors"][mask],
        "labels": data_dict["labels"][mask]
    }

    if len(filtered_scan["points"]) == 0:
        return data_dict  # No points with this label

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_scan["points"])
    clusters = np.array(pcd.cluster_dbscan(eps=dbscan_epsilon, min_points=dbscan_min_points, print_progress=False))

    if clusters.max() < 0:  # No clusters found
        return data_dict

    for cluster_id in range(0, clusters.max() + 1):
        cluster_mask = clusters == cluster_id
        # Generate new intensity values for this cluster
        if use_two_distributions:
            is_light = np.random.rand() < light_prob
            if is_light:
                new_intensity_values = sample_truncated_normal(mean_light, std_dev_light, cluster_mask.sum())
            else:
                new_intensity_values = sample_truncated_normal(mean_dark, std_dev_dark, cluster_mask.sum())
        else:
            # Use single distribution
            new_intensity_values = sample_truncated_normal(mean, std_dev, cluster_mask.sum())

        # Update the intensity values in filtered_scan
        filtered_scan["colors"][cluster_mask] = new_intensity_values.reshape(-1, 1)

    data_dict["colors"][mask] = filtered_scan["colors"]
    return data_dict
