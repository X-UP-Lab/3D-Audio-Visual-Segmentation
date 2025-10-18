import os
import numpy as np
from sklearn.cluster import DBSCAN


def run_dbscan_clustering(target_object_grid_points, eps=0.04, min_samples=6):
    """
    Runs DBSCAN clustering and returns only large clusters with volume > mean + 0.5 * std.
    """
    if target_object_grid_points.size == 0:
        return None

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(target_object_grid_points)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    cluster_volumes = {}
    for k in set(labels):
        if k != -1:
            class_member_mask = (labels == k)
            cluster_volumes[k] = np.sum(class_member_mask)

    if not cluster_volumes:
        return None

    mean_volume = np.mean(list(cluster_volumes.values()))
    std_volume = np.std(list(cluster_volumes.values()))
    threshold = mean_volume + 0.5 * std_volume

    large_clusters = [cluster for cluster, volume in sorted(cluster_volumes.items(), key=lambda x: x[1], reverse=True) if volume > threshold]
    large_cluster_masks = {cluster: (labels == cluster) for cluster in large_clusters}
    return large_cluster_masks


def get_distance(audio_intensity_centre: np.ndarray, cluster_proposal: np.ndarray):
    """
    Returns Euclidean distance between cluster centre and audio intensity centre.
    """
    distance = np.linalg.norm(np.mean(cluster_proposal[:, [0, 2]], axis=0) - audio_intensity_centre[[0, 2]])
    return distance


def audio_informed_refiner(large_cluster_masks, target_object_grid_points, audio_intensity_av):
    """
    Returns large cluster closest to the audio intensity centre.
    """
    if not large_cluster_masks or audio_intensity_av is None:
        return None

    min_distance = float('inf')
    chosen_mask = None
    for mask_index in large_cluster_masks.keys():
        cluster_proposal = target_object_grid_points[large_cluster_masks[mask_index]]
        distance = get_distance(audio_intensity_av, cluster_proposal)
        if distance < min_distance:
            min_distance = distance
            chosen_mask = large_cluster_masks[mask_index]
    return chosen_mask