import os
from pathlib import Path
from datetime import time
from typing import Optional
from moviepy import VideoFileClip
from utils.time import parse_time

def cut_clip(input_video_path:Path, start_time:Optional[float], end_time:Optional[float], output_video_path:Path):
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    clip = VideoFileClip(input_video_path)

    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = clip.duration

    if start_time == 0.0 and end_time == clip.duration:
        new_clip = clip
    else:
        new_clip = clip.subclipped(start_time, end_time)

    new_clip.write_videofile(output_video_path, codec="libx264")
    clip.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Cut a video clip from a larger video file.')
    parser.add_argument('input_video', type=str, help='Path to the input video file.')
    parser.add_argument('output_video', type=str, help='Path to save the output video clip.')
    parser.add_argument('--start_time', type=str, default=None, help='Start time in seconds (default: beginning of the video).')
    parser.add_argument('--end_time', type=str, default=None, help='End time in seconds (default: end of the video).')

    args = parser.parse_args()

    start_time = parse_time(args.start_time) if args.start_time else None
    end_time = parse_time(args.end_time) if args.end_time else None

    cut_clip(args.input_video, start_time, end_time, args.output_video)
    