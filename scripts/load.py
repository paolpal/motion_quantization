from pats.utils import load_multiple_samples, get_speaker_intervals
from pathlib import Path

data_root = Path('/home/ubuntu/palumbo/Posemi/dataset/data/pats/data')

intervals = get_speaker_intervals(speaker='fallon', split='dev', data_root=data_root)
clips = load_multiple_samples(speaker='fallon', interval_ids=intervals, data_root=data_root)

