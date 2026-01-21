import torch
from torch.utils.data import Dataset

import os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from pats.utils import Skeleton2D
from pats.utils import load_multiple_samples, get_speaker_intervals

from typing import List, Optional, Union, Literal

class PoseDataset(Dataset):
    def __init__(self, speakers: List[str],
                 data_root: Union[str, Path],
                 split: Literal["train", "dev", "test"] = "train", 
                 cache_dir: Optional[Union[str, Path]] = None, 
                 force_rebuild: bool = False
                 ):
        self.speakers = sorted(speakers)
        self.data_root = Path(data_root)
        
        if cache_dir is not None:
            if isinstance(cache_dir, str):
                cache_dir = Path(cache_dir)
            spk_hash = "_".join(self.speakers[:3]) + f"_etc_{len(self.speakers)}"
            self.cache_file = cache_dir / f"{spk_hash}_{split}.pt"
        else:
            self.cache_file = None
        
        if (
            self.cache_file is not None and
            self.cache_file.exists() and
            not force_rebuild
        ):
            print(f"Caricamento cache: {self.cache_file}")
            data_dict = torch.load(self.cache_file, weights_only=False)
            self.clips = data_dict['clips']
            self.clip_indices = data_dict['clip_indices']
            self.all_frames = torch.cat(self.clips, dim=0)
        else:
            self._build_and_save(split)

    def _build_and_save(self, split: Literal["train", "dev", "test"]):
        print(f"Generazione cache per {len(self.speakers)} speaker...")
        all_clips = []
        clip_mapping = []
        global_id = 0

        # Caricamento parallelo degli speaker
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            
            futures = [executor.submit(self._load_speaker_clips, s, split) for s in self.speakers]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Speakers"):
                clips = future.result()
                for c in clips:
                    c_tensor = torch.tensor(c, dtype=torch.float32)
                    all_clips.append(c_tensor)
                    clip_mapping.extend([global_id] * c_tensor.shape[0])
                    global_id += 1

        # Concatenazione temporanea per normalizzazione globale
        temp_all = torch.cat(all_clips, dim=0)
        temp_all = torch.tensor(Skeleton2D.normalize_skeleton(temp_all.numpy()))

        # Ricostruzione clip normalizzate
        start = 0
        normalized_clips = []
        for c in all_clips:
            end = start + len(c)
            normalized_clips.append(temp_all[start:end])
            start = end
            
        self.clips = normalized_clips
        self.clip_indices = np.array(clip_mapping)
        self.all_frames = temp_all

        if self.cache_file is not None:
            torch.save({'clips': self.clips, 'clip_indices': self.clip_indices}, self.cache_file)

    def _load_speaker_clips(self, speaker: str, split: Literal["train", "dev", "test"]):
        intervals = get_speaker_intervals(speaker=speaker, split=split, data_root=self.data_root)
        samples = load_multiple_samples(speaker=speaker, interval_ids=intervals, data_root=self.data_root)
        processed = []
        for s in samples:
            p = s['pose']
            p[:, 0] = [0.0, 0.0] # Centering
            processed.append(p)
        return processed
    
    def get_clip_indices(self):
        return np.unique(self.clip_indices)
    
    def get_clip_by_index(self, index: int):
        return self.clips[index]

    def __len__(self) -> int: return len(self.all_frames)
    def __getitem__(self, idx: int): return self.all_frames[idx]