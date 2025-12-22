from typing import Optional
from hdbscan import HDBSCAN
import numpy as np
from quantization.codebook import PoseCodebook
from sklearn.discriminant_analysis import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
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
    transformer = UMAP(n_components=2)
    print(f"Poses shape{data.shape}")
    scaled : np.ndarray = transformer.fit_transform(data.reshape(len(data), -1)) # type: ignore
    print(f"Transformed poses shape: {scaled.shape} {type(scaled)}")

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
        cluster_skeletons = data[mask]
        if len(cluster_points) > 0:
            center = clusterer.weighted_cluster_medoid(cluster_id)
            #center = cluster_points.mean(axis=0)
            print(f"Cluster {cluster_id}: size={len(cluster_points)}, center shape={center.shape}, center={center}")
            idx = np.argmin(np.linalg.norm(cluster_points - center, axis=1))
            #pose = transformer.inverse_transform(center.reshape(1, -1))
            #pose = cluster_skeletons.mean(axis=0)
            pose = cluster_skeletons[idx]
            pose = pose.reshape(-1, 2)
            posemids.append(pose)
            centers.append(center)
            sizes.append(len(cluster_points))

    # cluster_plot(scaled, labels, centroids=np.array(centers))
    cluster_plot(scaled, labels)

    if len(centers) == 0:
        return None

    return PoseCodebook(clusterer=clusterer, centroids=np.array(centers), transformer=transformer, poses=posemids)
