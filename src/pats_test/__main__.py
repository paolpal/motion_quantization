from pathlib import Path
import numpy as np
from pats.utils import load_multiple_samples, get_speaker_intervals
from quantization.clustering import cluster_poses
from utils.plot import plot_codebook
from quantization.codebook import PoseCodebook
from utils.skeletonPATS import SkeletonPATS

if __name__ == "__main__":
    # Esempio di caricamento di più campioni dal dataset PATS
    speaker = "seth"
    split = "dev"
    data_path = Path(__file__).resolve().parents[2]/"data"

    intervals = get_speaker_intervals(speaker, split=split)
    clips = load_multiple_samples(speaker=speaker, interval_ids=intervals)

    skeleton_poses = []
    for clip in clips:

        pose = clip['pose']

        ########## filtering ##########

        print(f"Original pose shape: {pose.shape}")

        #filtered_poses = SkeletonPATS.filter_skeleton(pose, remove_head=False)
        pose[:,0] = [0.0, 0.0]
        normal_poses = SkeletonPATS.normalize_skeleton(pose)
        ###############################
        skeleton_poses.append(normal_poses)

    skeleton_poses = np.concatenate(skeleton_poses, axis=0)
    print(f"Loaded {skeleton_poses.shape} poses for speaker '{speaker}' in split '{split}'.")

    # min_cluster_size = int(skeleton_poses.shape[0]*(0.5/100))
    min_cluster_size = 20

    polar_poses = SkeletonPATS.encode_as_polar(skeleton_poses)

    codebook = cluster_poses(polar_poses, min_cluster_size=min_cluster_size, min_samples=min_cluster_size, n_components=16)
    #codebook = cluster_poses(skeleton_poses, min_cluster_size=min_cluster_size, min_samples=min_cluster_size)

    if codebook is not None:
        print(f"Clustering completed. Found {len(codebook.centroids)} centroids.")
    else:
        print("Clustering returned no codebook.")
        exit(-1)

    codebook.poses = [SkeletonPATS.decode_from_polar(pose) for pose in codebook.poses]

    plot_codebook(codebook, title="Pose Codebook", polar=True)

    # codebook_path = data_path / "datasets" / "codebook_pats.pkl"
    # codebook_path.parent.mkdir(parents=True, exist_ok=True)
    # codebook.save(codebook_path)
    
