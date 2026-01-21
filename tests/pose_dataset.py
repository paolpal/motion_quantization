from motion_quantization.dataset import PoseDataset
from random import randint


dataset = PoseDataset(
    speakers=['fallon'],
    split="dev",
    data_root="/home/paolo/Projects/Gesture/pats/data",
    cache_dir="cache"
)


print(f"Dataset length: {len(dataset)}")

clips_idx = dataset.get_clip_indices()

print(f"Number of clips: {len(clips_idx)}")

max_clips = len(clips_idx)
rand_idx = randint(0, max_clips-1)
print(f"Random index: {rand_idx}")

clip = dataset.get_clip_by_index(rand_idx)
print(f"Clip shape: {clip.shape}")