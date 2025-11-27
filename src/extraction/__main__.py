import argparse
from extraction.pose import extract_pose
from extraction.transcription import fast_transcribe
from extraction.audio import extract_audio
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process clips to extract pose and transcription.')
    parser.add_argument('--data_path', required=True, help='Path to the data folder')
    parser.add_argument('--speaker', required=False, help='Speaker name to process clips for')
    parser.add_argument('--model_size_pose', default='yolov8n-pose.pt', help='Size of the YOLO pose model to use (default: yolov8n-pose.pt)')
    parser.add_argument('--model_size_transcription', default='tiny', help='Size of the Whisper model to use (default: tiny)')
    
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument('--train', action='store_true', help='Process train dataset')
    split.add_argument('--dev', action='store_true', help='Process dev dataset')
    split.add_argument('--test', action='store_true', help='Process test dataset')
    
    args = parser.parse_args()

    # list file in folder
    data_path = Path(args.data_path)
    clips_folder = data_path / "clips"
    split_name = 'train' if args.train else 'dev' if args.dev else 'test'
    split_folder = clips_folder / split_name 
    speaker_folder = split_folder / args.speaker if args.speaker else split_folder

    video_files = list(speaker_folder.rglob("*.mp4"))

    print(f"Processing {len(video_files)} video files for pose extraction and transcription...")

    for clip in video_files:
        output_folder = data_path / "processed" / split_name / clip.parent.stem / clip.stem
        print(f"Processing {clip}...")

        # Extract pose
        extract_pose(clip, output_folder, model_size=args.model_size_pose)
        # Transcribe audio

        wave_file = extract_audio(clip, output_folder)
        fast_transcribe(wave_file, output_folder, model_size=args.model_size_transcription)
    

