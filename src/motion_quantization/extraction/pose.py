import json
import os
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO # type: ignore
import numpy as np

def extract_pose(video_file: Path, output_folder: Path, model_size: str = "yolov8n-pose.pt") -> np.ndarray:
    """
    Extracts pose keypoints from the given video file using OpenPose.

    Args:
        video_file (Path): Path to the input video file.
        output_folder (Path): Path to the folder where extracted keypoints will be saved.

    Returns:
        keypoints_data (np.ndarray): A numpy array containing the extracted keypoints for each frame. The shape of the array is (num_frames, num_keypoints, 2).
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

        frame_keypoints = np.array(frame_keypoints)
        frame_keypoints = np.squeeze(frame_keypoints)
        keypoints_data.append(frame_keypoints)
    
    keypoints_tensor: np.ndarray = np.array(keypoints_data)
    keypoints_tensor = np.squeeze(keypoints_tensor)

    print(keypoints_tensor.shape)

    cap.release()

    with open(output_json, "w") as f:
        data = {
            "keypoints": keypoints_tensor.tolist(),
            "video_metadata": {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames
            }
        }
        json.dump(data, f, indent=2)


    return keypoints_tensor

    