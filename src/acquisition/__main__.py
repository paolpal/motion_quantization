import argparse
from pathlib import Path
from acquisition.clips import cut_clip
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.time import parse_time, strip_date
from utils.youtube import check
from src.acquisition.download import download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download videos using yt-dlp')
    parser.add_argument('--speaker', required=False, help='Speaker name to download videos for')
    parser.add_argument('--data_path', required=True, help='Path to the data folder')
    parser.add_argument('--intervals_file', default='cmu_intervals_df.csv', help='Path to the intervals CSV file (default: data/cmu_intervals_df.csv)')
    
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument('--no-download', action='store_true', help='Flag to skip downloading videos and only cut clips from existing videos')
    phase_group.add_argument('--no-cut', action='store_true', help='Flag to skip cutting clips and only download videos')

    # Gruppo mutualmente esclusivo per dataset
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument('--test', action='store_true', help='Flag to indicate if test data should be downloaded')
    dataset_group.add_argument('--dev', action='store_true', help='Flag to indicate if dev data should be downloaded')
    dataset_group.add_argument('--train', action='store_true', help='Flag to indicate if train data should be downloaded')
    dataset_group.add_argument('--all', action='store_true', help='Flag to indicate if all datasets should be downloaded')

    args = parser.parse_args()    

    data_path = Path(args.data_path)
    intervals_file = data_path / args.intervals_file

    df = pd.read_csv(intervals_file)

    if args.speaker:
        df = df[df['speaker'] == args.speaker]
    
    # Determina quale dataset scaricare (default: train)
    if args.test:
        df = df[df['dataset'] == 'test']
    elif args.dev:
        df = df[df['dataset'] == 'dev']
    elif args.train:
        df = df[df['dataset'] == 'train']
    elif args.all:
        pass  # Keep all datasets
    else:
        df = df[df['dataset'] == 'train']

    tqdm.pandas()
    df['video_id'] = df['video_link'].apply(lambda x: x.split('=')[-1])
    df['available'] = df['video_link'].progress_apply(check)
    df = df[df['available'] == True]

    df_download = df.drop_duplicates(subset=['video_link'])

    print(f"Starting download of {len(df_download)} videos...")


    if not args.no_download:
        Parallel(n_jobs=-1)(  # Adjust n_jobs based on your CPU cores
            delayed(download)(
                row['video_link'],
                data_path / 'raw' / row['speaker'],
                f"{row['speaker']}_{row['video_id']}"
            ) for _, row in tqdm(df_download.iterrows(), total=len(df_download))
        )

    print(f"Start clip cutting... Total clips to cut: {len(df)}")
    df['start_time'] = df['start_time'].apply(strip_date)
    df['end_time'] = df['end_time'].apply(strip_date)

    if not args.no_cut:
        Parallel(n_jobs=-1)(  # Adjust n_jobs based on your CPU cores
            delayed(cut_clip)(
                data_path / 'raw' / row['speaker'] / f"{row['speaker']}_{row['video_id']}.mp4",
                parse_time(row['start_time']),
                parse_time(row['end_time']),
                data_path / 'clips' / row['dataset'] / row['speaker'] / f"{row['speaker']}_{row['video_id']}_{row['start_time'].replace(':', '-')}_{row['end_time'].replace(':', '-')}.mp4"
            ) for _, row in tqdm(df.iterrows(), total=len(df))
        )

