import argparse
import json
from pathlib import Path
import numpy as np
from quantization.normalization import filter_torso, normalize_pose
from quantization.clustering import cluster_poses
from sklearn.discriminant_analysis import StandardScaler
from tqdm import tqdm
from utils.plot import plot_codebook, plot_skeleton

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantization of pose data.')
    parser.add_argument('--data_path', required=True, help='Path to the data folder')
    parser.add_argument('--speaker', required=False, help='Speaker name')
    parser.add_argument('--min_cluster_size', type=int, default=20, help='Minimum cluster size for HDBSCAN')
    
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument('--train', action='store_true', help='Process train dataset')
    split.add_argument('--dev', action='store_true', help='Process dev dataset')
    split.add_argument('--test', action='store_true', help='Process test dataset')
    
    args = parser.parse_args()

    data_path = Path(args.data_path)
    split_name = 'train' if args.train else 'dev' if args.dev else 'test'
    processed_folder = data_path / "processed" / split_name
    speaker_folder = processed_folder / args.speaker if args.speaker else processed_folder

    # list all *_keypoints.json files in folder and subfolders
    keypoint_files = list(speaker_folder.rglob("*_keypoints.json"))
    print(f"Found {len(keypoint_files)} keypoint files to quantize.")

    all_poses = []
    for keypoint_file in keypoint_files:
        with open(keypoint_file, 'r') as f:
            poses = json.load(f)
        for frame in tqdm(poses["keypoints_data"], desc=f"Processing {keypoint_file.name}"):
            if 'keypoints' in frame and frame['keypoints']:
                kp = frame['keypoints'][0] if isinstance(frame['keypoints'], list) else frame['keypoints']
                # Handle nested structure: keypoints might be [[[x,y], ...]] instead of [[x,y], ...]
                if isinstance(kp, list) and len(kp) > 0 and isinstance(kp[0], list) and len(kp[0]) > 0 and isinstance(kp[0][0], list):
                    kp = kp[0]  # Extract the actual keypoints from the extra nesting
                kp = filter_torso(kp, remove_head=True)
                normalized = normalize_pose(kp)
                if normalized is not None:
                    all_poses.append(normalized)

    # Forse posso proiettare in uno spazio più basso prima di clusterizzare
    # tipo spazio latente con PCA o VQ-VAE

    print(f"Total normalized poses collected: {len(all_poses)}")
    print(f"{type(all_poses[0])} with shape {all_poses[0].shape}")

    codebook = cluster_poses(all_poses, min_cluster_size=args.min_cluster_size)

    if codebook is not None:
        print(f"Clustering completed. Found {len(codebook.centroids)} centroids.")
    else:
        print("Clustering returned no codebook.")
        exit(-1)

    plot_codebook(codebook, title="Pose Codebook")

    codebook_path = data_path / "datasets" / "codebook.pkl"
    codebook_path.parent.mkdir(parents=True, exist_ok=True)
    codebook.save(codebook_path)

    


    