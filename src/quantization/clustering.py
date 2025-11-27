from typing import Optional
from hdbscan import HDBSCAN
import numpy as np
from quantization.codebook import PoseCodebook
from sklearn.discriminant_analysis import StandardScaler
from sklearn.decomposition import PCA

def cluster_poses(poses, min_cluster_size=10) -> Optional[PoseCodebook]:
    """
    poses: list of np.array, each of shape (num_keypoints, 2)
    min_cluster_size: minimum size of clusters for HDBSCAN
    Returns: list of cluster labels and cluster centers
    """

    if len(poses) == 0:
        return None

    data = np.array(poses)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
    labels = clusterer.fit_predict(scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters == labels.max() + 1

    centers = []
    sizes = []
    posemids = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_points = scaled[mask]
        if len(cluster_points) > 0:
            center = clusterer.weighted_cluster_medoid(cluster_id)
            pose = scaler.inverse_transform(center.reshape(1, -1))
            pose = pose.reshape(-1, 2)
            posemids.append(pose)
            centers.append(center)
            sizes.append(len(cluster_points))

    return PoseCodebook(clusterer=clusterer, centroids=np.array(centers), scaler=scaler, poses=posemids)
