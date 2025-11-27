import argparse
import json
from pathlib import Path

from quantization.codebook import PoseCodebook
from quantization.normalization import filter_torso, normalize_pose
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset builder for quantized pose sequences.')
    parser.add_argument('--data_path', required=True, help='Path to the data folder')
    parser.add_argument('--speaker', required=False, help='Speaker name')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for the output sequences')
    
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument('--train', action='store_true', help='Process train dataset')
    split.add_argument('--dev', action='store_true', help='Process dev dataset')
    split.add_argument('--test', action='store_true', help='Process test dataset')

    args = parser.parse_args()

    data_path = Path(args.data_path)
    split_name = 'train' if args.train else 'dev' if args.dev else 'test'

    dataset_folder = data_path / "datasets" 
    split_folder = dataset_folder / split_name
    speaker_folder = split_folder / args.speaker if args.speaker else split_folder

    codebook = PoseCodebook.load(dataset_folder / "codebook.pkl")

    # list all folder containing *_keypoints.json and corresponding *_segments.json
    from dataset.builder import build
    processed_folder = data_path / "processed" / split_name
    speaker_processed_folder = processed_folder / args.speaker if args.speaker else processed_folder

    keypoint_files = list(speaker_processed_folder.rglob("*_keypoints.json"))
    print(f"Found {len(keypoint_files)} keypoint files to build dataset.")

    datasets_files = []
    for keypoint_file in keypoint_files:
        segment_file = keypoint_file.with_name(keypoint_file.stem.replace("_keypoints", "_segments") + ".json")
        if not segment_file.exists():
            print(f"Segment file {segment_file} not found, skipping.")
            continue

        poses = []
        with open(keypoint_file, 'r') as f:
            poses_dict = json.load(f)
        for frame in tqdm(poses_dict["keypoints_data"], desc=f"Processing {keypoint_file.name}"):
            if 'keypoints' in frame and frame['keypoints']:
                kp = frame['keypoints'][0] if isinstance(frame['keypoints'], list) else frame['keypoints']
                # Handle nested structure: keypoints might be [[[x,y], ...]] instead of [[x,y], ...]
                if isinstance(kp, list) and len(kp) > 0 and isinstance(kp[0], list) and len(kp[0]) > 0 and isinstance(kp[0][0], list):
                    kp = kp[0]  # Extract the actual keypoints from the extra nesting
                kp = filter_torso(kp, remove_head=True)
                normalized = normalize_pose(kp)
                if normalized is not None:
                    poses.append(normalized)
        with open(segment_file, 'r') as f:
            segments = json.load(f)

        output_folder = speaker_folder / keypoint_file.parent.relative_to(speaker_processed_folder)
        print(f"Building dataset for {keypoint_file.parent.name} into {output_folder}")

        new_dataset = build(poses, segments, codebook, output_folder, fps=poses_dict["video_metadata"]["fps"])
        datasets_files.append(new_dataset)

    print("Datasets built:")
    for dataset_file in datasets_files:
        print(f" - {dataset_file}")

    # Concatenate all jsonl files into a single one for the split
    combined_jsonl = speaker_folder / f"{split_name}_dataset.jsonl"
    with open(combined_jsonl, 'w', encoding='utf-8') as outfile:
        for dataset_file in datasets_files:
            with open(dataset_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)