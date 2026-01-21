import json
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel


def fast_transcribe(wave_file: Path, output_folder: Path, model_size: str = "tiny") -> tuple[str, list]:
    """
    Transcribes the given audio file using the Whisper model with faster settings.

    Args:
        wave_file (Path): Path to the input WAV audio file.
        output_folder (Path): Path to the output folder.
        model_size (str): Size of the Whisper model to use (default is "tiny").

    Returns:
        str: The transcribed text.
    """


    output_folder.mkdir(parents=True, exist_ok=True)
    transcript_path = output_folder / f"{wave_file.stem}_transcript.txt"
    segments_path = output_folder / f"{wave_file.stem}_segments.json"

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
                    "end": word.end
                })

        segments_data.append(seg_data)

        transcription += seg.text + " "

    transcription_str = str(transcription) if not isinstance(transcription, str) else transcription

    with open(transcript_path, "w") as f:
        f.write(transcription_str)

    with open(segments_path, "w") as f:
        json.dump(segments_data, f, indent=2)

    return transcription_str.strip(), segments_data
