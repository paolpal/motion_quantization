from typing import Optional
from hdbscan import HDBSCAN
import numpy as np
from quantization.codebook import PoseCodebook
from sklearn.discriminant_analysis import StandardScaler
from sklearn.decomposition import PCA
from utils.cluster_plot import cluster_plot
from sklearn.cluster import DBSCAN

def cluster_poses(poses: np.ndarray, min_cluster_size=10, min_samples=5) -> Optional[PoseCodebook]:
    """
    poses: list of np.array, each of shape (num_keypoints, 2)
    min_cluster_size: minimum size of clusters for HDBSCAN
    Returns: list of cluster labels and cluster centers
    """

    if len(poses) == 0:
        return None

    data = np.array(poses)

    transformer = PCA(n_components=2)
    print(f"Fitting transformer on data of shape {data.shape}")
    scaled = transformer.fit_transform(data.reshape(len(data), -1))
    print(f"Transformed data shape: {scaled.shape}")

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=True)
    #clusterer = DBSCAN(eps=0.1, min_samples=min_cluster_size)
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
            #center = clusterer.weighted_cluster_medoid(cluster_id)
            center = cluster_points.mean(axis=0)
            print(f"Cluster {cluster_id}: size={len(cluster_points)}, center shape={center.shape}, center={center}")
            pose = transformer.inverse_transform(center.reshape(1, -1))
            pose = pose.reshape(-1, 2)
            posemids.append(pose)
            centers.append(center)
            sizes.append(len(cluster_points))

    cluster_plot(scaled, labels, centroids=np.array(centers))

    return PoseCodebook(clusterer=clusterer, centroids=np.array(centers), transformer=transformer, poses=posemids)
