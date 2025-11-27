import os
import sys
import yt_dlp
import logging
from contextlib import redirect_stdout, redirect_stderr

logger = logging.getLogger(__name__)

def download(link:str, output_dir:str, file_name:str) -> None:
    """Download a video from a given link using yt-dlp.

    Args:
        link (str): The URL of the video to download.
        output_dir (str): The directory where the video will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'outtmpl': f'{output_dir}/{file_name}.%(ext)s',
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',  # Download best quality video and audio
        'quiet': True,  # Suppress verbose output
    }

    with open(os.devnull, 'w') as null_device:
        with redirect_stdout(null_device), redirect_stderr(null_device):
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: # pyright: ignore[reportArgumentType]
                ydl.download([link])
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download a video using yt-dlp.')
    parser.add_argument('link', type=str, help='The URL of the video to download.')
    parser.add_argument('output_dir', type=str, help='The directory where the video will be saved.')
    parser.add_argument('file_name', type=str, help='The name of the output video file (without extension).')

    args = parser.parse_args()

    download(args.link, args.output_dir, args.file_name)
