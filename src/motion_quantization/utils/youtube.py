from functools import lru_cache
import os
import yt_dlp
from contextlib import redirect_stdout, redirect_stderr

@lru_cache(maxsize=128)
def check(url: str) -> bool:
    """Check if a YouTube video is available.

    Args:
        url (str): The URL of the YouTube video.
    Returns:
        bool: True if the video is available, False otherwise.
    """
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
    }

    try:
        with open(os.devnull, 'w') as null_device:
            with redirect_stdout(null_device), redirect_stderr(null_device):
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # pyright: ignore[reportArgumentType]
                    info = ydl.extract_info(url, download=False)
                    return not info.get('is_live', False)  # Check if the video is live
    except Exception as e:
        # print(f"Error checking video {url}: {e}")
        return False