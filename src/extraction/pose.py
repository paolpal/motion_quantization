import json
import os
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO # type: ignore

def extract_pose(video_file: Path, output_folder: Path, model_size: str = "yolov8n-pose.pt") -> list:
    """
    Extracts pose keypoints from the given video file using OpenPose.

    Args:
        video_file (Path): Path to the input video file.
        output_folder (Path): Path to the folder where extracted keypoints will be saved.

    Returns:
        None
    """

    os.makedirs(output_folder, exist_ok=True)
    output_json = output_folder / f"{video_file.stem}_keypoints.json"

    model = YOLO(model_size, verbose=False)

    cap = cv2.VideoCapture(str(video_file))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    keypoints_data = []

    for frame_idx in tqdm(range(total_frames), desc="Extracting Pose", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        frame_keypoints = []
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.tolist()
                frame_keypoints.append(keypoints)

        keypoints_data.append({
            "frame_index": frame_idx,
            "keypoints": frame_keypoints
        })

    cap.release()

    with open(output_json, "w") as f:
        data = {
            "keypoints_data": keypoints_data,
            "video_metadata": {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames
            }
        }
        json.dump(data, f, indent=2)


    return keypoints_data

    