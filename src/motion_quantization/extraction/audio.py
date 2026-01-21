from pathlib import Path

from moviepy import VideoFileClip

def extract_audio(clip_path: Path, output_folder: Path) -> Path:
    """
    Extracts audio from a video file and saves it as a WAV file.

    Args:
        clip_path (Path): Path to the input video file.
        processed (Path): Path to the output folder.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    wav_path = output_folder / f"{clip_path.stem}.wav"

    clip = VideoFileClip(clip_path)
    if clip.audio is not None:
        clip.audio.write_audiofile(
            wav_path,
            fps=16000,
            codec="pcm_s16le",
            logger=None
        )
    clip.close()
    return wav_path


   