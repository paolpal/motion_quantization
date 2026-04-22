import json
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel
from typing import Optional


def fast_transcribe(wave_file: Path, output_folder: Optional[Path] = None, model_size: str = "tiny", fps: int = 15) -> tuple[str, list]:
    """
    Transcribes the given audio file using the Whisper model with faster settings.

    Args:
        wave_file (Path): Path to the input WAV audio file.
        output_folder (Path): Path to the output folder.
        model_size (str): Size of the Whisper model to use (default is "tiny").
        fps (int): Frames per second for frame-level timestamping (default is 15).

    Returns:
        str: The transcribed text.
    """

    model = WhisperModel(model_size, device="auto")
    segments, info = model.transcribe(str(wave_file), word_timestamps=True)

    transcription = ""
    segments_data = []

    for seg in tqdm(segments, desc="Transcribing", unit="segment"):
        seg_data = {
                "text": seg.text.strip(),
                "start": seg.start,
                "end": seg.end,
                "words": []
            }

        if hasattr(seg, 'words') and seg.words:
            for word in seg.words:
                seg_data["words"].append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end,
                    "start_frame": int(round(word.start * fps)),
                    "end_frame": int(round(word.end * fps)),
                })

        segments_data.append(seg_data)

        transcription += seg.text + " "

    transcription_str = str(transcription) if not isinstance(transcription, str) else transcription

    if output_folder is not None:
        output_folder.mkdir(parents=True, exist_ok=True)
        transcript_path = output_folder / f"{wave_file.stem}_transcript.txt"
        segments_path = output_folder / f"{wave_file.stem}_segments.json"

        with open(transcript_path, "w") as f:
            f.write(transcription_str)

        with open(segments_path, "w") as f:
            json.dump(segments_data, f, indent=2)

    return transcription_str.strip(), segments_data

def iter_transcribe(wave_file: Path, model_size: str = "tiny", fps: int = 15):
    """
    Transcribes the given audio file using the Whisper model with faster settings, yielding segments one by one.

    Args:
        wave_file (Path): Path to the input WAV audio file.
        model_size (str): Size of the Whisper model to use (default is "tiny").
        fps (int): Frames per second for frame-level timestamping (default is 15).
    """
    model = WhisperModel(model_size, device="auto")
    segments, info = model.transcribe(str(wave_file), word_timestamps=True)

    for seg in tqdm(segments, desc="Transcribing", unit="segment"):
        seg_data = {
                "text": seg.text.strip(),
                "start": seg.start,
                "end": seg.end,
                "words": []
            }

        if hasattr(seg, 'words') and seg.words:
            for word in seg.words:
                seg_data["words"].append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end,
                    "start_frame": int(round(word.start * fps)),
                    "end_frame": int(round(word.end * fps)),
                })

        yield seg_data