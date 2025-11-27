from pathlib import Path
import numpy as np
from hdbscan import HDBSCAN
from hdbscan.prediction import approximate_predict
import pickle

from sklearn.discriminant_analysis import StandardScaler
from quantization.normalization import normalize_pose


class PoseCodebook:
    """
    A simple vector quantization codebook for poses.
    
    Parameters
    ----------
    centroids : array-like, shape (n_clusters, d)
        The representative vectors (centroids or medoids) for each cluster.
    tokens : list or array-like, optional
        Optional token IDs associated with each centroid. 
        If None, tokens = range(n_clusters).
    """

    def __init__(self, clusterer: HDBSCAN, centroids, scaler, poses, tokens=None):
        self.clusterer: HDBSCAN = clusterer
        self.scaler : StandardScaler = scaler
        self.poses = poses

        assert len(poses) == len(centroids), "Number of poses must match number of centroids"

        centroids = np.asarray(centroids, dtype=np.float32)

        if centroids.ndim != 2:
            raise ValueError(f"centroids must have shape (n_clusters, vector_dim), got {centroids.shape}")

        self.centroids = centroids
        self.n_clusters = centroids.shape[0]

        if tokens is None:
            self.tokens = np.arange(self.n_clusters)
        else:
            tokens = np.asarray(tokens)
            if len(tokens) != self.n_clusters:
                raise ValueError("tokens must have same length as centroids")
            self.tokens = tokens

    def predict(self, pose: list[np.ndarray] | np.ndarray) -> int | np.ndarray:
        """
        Assign the closest centroid to each pose vector.
        
        Parameters
        ----------
        pose : array-like, shape (n,2) or (batch, n, 2) 
            Single pose vector or batch of pose vectors.
        
        Returns
        -------
        cluster_ids : int or ndarray of shape (batch,)
            The index of the closest centroid for each input vector.
            Returns -1 if the pose cannot be normalized.
        """
        # Normalizza la posa
        if isinstance(pose, list):
            norm_pose = []
            invalid_indices = []
            for i, p in enumerate(pose):
                n_p = normalize_pose(p)
                if n_p is None:
                    # Segna come invalido e usa placeholder
                    invalid_indices.append(i)
                    norm_pose.append(np.zeros(16))
                    print(f"Pose {i} is invalid (None from normalize)")
                else:
                    norm_pose.append(n_p)
            norm_pose = np.array(norm_pose)
            
            print(f"Normalized {len(norm_pose)} poses, {len(invalid_indices)} invalid")
            print(f"Norm pose shape: {norm_pose.shape}")
            
            # Se tutte le pose sono invalide
            if len(invalid_indices) == len(pose):
                print("All poses are invalid!")
                return np.full(len(pose), -1)
        else:
            norm_pose = normalize_pose(pose)
            if norm_pose is None:
                print("Single pose is invalid (None from normalize)")
                return -1
            invalid_indices = []
            print(f"Single pose normalized: {norm_pose.shape}")
        
        print(f"About to scale: {norm_pose.shape}")
        # Scala i valori normalizzati
        scaled_pose = self.scaler.transform(norm_pose)
        print(f"Scaled pose shape: {scaled_pose.shape}")
        
        # Predici il cluster usando nearest centroid (più robusto di approximate_predict)
        # Calcola la distanza euclidea da tutti i centroids
        # scaled_pose shape: (batch, 16), centroids shape: (n_clusters, 16)
        distances = np.linalg.norm(scaled_pose[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        cluster_ids = np.argmin(distances, axis=1)
        print(f"Cluster IDs from nearest centroid: {cluster_ids[:20]}...")  # Mostra i primi 20
        
        # Marca come -1 le pose invalide
        if isinstance(pose, list) and invalid_indices:
            cluster_ids = cluster_ids.astype(int)
            for idx in invalid_indices:
                cluster_ids[idx] = -1
            print(f"After marking invalids: {cluster_ids[:20]}...")
        
        
        # Restituisci un singolo valore se è una singola posa
        if len(cluster_ids) == 1:
            return cluster_ids[0]
        return cluster_ids

        

    def __len__(self):
        """Return number of codebook entries."""
        return self.n_clusters

    def __repr__(self):
        return f"PoseCodebook(n_clusters={self.n_clusters}, dim={self.centroids.shape[1]})"

    def save(self, filepath: str|Path):
        """Save the codebook to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath) -> 'PoseCodebook':
        """Load a codebook from a file."""
        with open(filepath, 'rb') as f:
            codebook = pickle.load(f)
        return codebook
        