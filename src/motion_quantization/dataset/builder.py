import json
import os
from pathlib import Path

import numpy as np
from motion_quantization.quantization import PoseTokenizer
import torch


def build(poses:list[np.ndarray], segments:list, codebook:PoseTokenizer, output_folder:Path, fps=60) -> Path:
    """
    Build a dataset of quantized pose sequences.

    Parameters
    ----------
    poses : list of np.array
        List of pose arrays, each of shape (num_keypoints, 2).
    segments : list of tuples
        List of (start_frame, end_frame) tuples defining segments.
    codebook : PoseTokenizer
        The pose codebook for quantization.
    output_folder : Path
        Folder to save the quantized sequences.
    fps : int
        Frames per second for the output sequences.

    Returns
    -------
    jsonl_path : Path
        Path to the generated JSONL dataset file.
    """

    output_folder.mkdir(parents=True, exist_ok=True)

    _, tokens = codebook.quantize(torch.tensor(poses, dtype=torch.float32))
    tokens = tokens.numpy()

    samples: list = []
    for segment in segments:
        text = segment['text'].strip()

        start_time = segment['start']
        end_time = segment['end']

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        segment_tokens = tokens[start_frame:end_frame]
        segment_tokens = segment_tokens.tolist()
        sample = {
            'text': text,
            'tokens': segment_tokens,
            'duration': end_time - start_time,
            'start': start_time,
            'end': end_time,
            'n_frames': len(segment_tokens)
        }
        if 'words' in segment and segment['words']:
            sample['words'] = segment['words']
        samples.append(sample)

    jsonl_path = output_folder / "dataset.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    return jsonl_path