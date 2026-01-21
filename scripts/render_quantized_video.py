import torch
import matplotlib.pyplot as plt
import numpy as np
from train.tuning import SkeletonVQVAE
from motion_quantization.utils.skeletonPATS import SkeletonPATS

from pats.utils import load_multiple_samples, get_speaker_intervals
from matplotlib.animation import FuncAnimation

def compare_quantized_animation(model, speaker="seth", clip_idx=30, interval=67, smooth_factor=0.5, output_filename=None):
    model.eval()
    device = next(model.parameters()).device

    # 1. Caricamento e Normalizzazione (Tuo codice invariato)
    print("Inizio confronto animazione...")
    print("Caricamento clip originale...")
    print(f"Ottenimento intervalli per speaker '{speaker}'...")
    intervals = get_speaker_intervals(speaker=speaker, split="train")
    print(f"Intervalli trovati: {len(intervals)}")
    print(f"Caricamento clip {clip_idx} per speaker '{speaker}'...")
    clips = load_multiple_samples(speaker=speaker, interval_ids=intervals[clip_idx:clip_idx+1])
    print("Clip caricata.")
    if not clips: 
        print("Errore: Nessuna clip trovata.")
        return

    raw_pose = clips[0]['pose']
    raw_pose[:, 0] = [0.0, 0.0]
    normalized_pose = SkeletonPATS.normalize_skeleton(raw_pose)
    parents = SkeletonPATS.parents()

    # 2. Generazione versione Quantizzata
    input_tensor = torch.tensor(normalized_pose, dtype=torch.float32).to(device)

    print("Generazione animazione quantizzata...")
    with torch.no_grad():
        x_recon, _, _ = model(input_tensor) 
        quantized_pose = x_recon.view(-1, 52, 2).cpu().numpy()

    # --- AGGIUNTA: INTERPOLAZIONE LINEARE (SMOOTHING) ---
    # Creiamo una copia per non sovrascrivere
    smoothed_pose = np.copy(quantized_pose)
    # Partiamo dal secondo frame e mediamo con il precedente
    for t in range(1, len(smoothed_pose)):
        smoothed_pose[t] = (1 - smooth_factor) * quantized_pose[t] + smooth_factor * smoothed_pose[t-1]
    # ----------------------------------------------------

    # 3. Setup della Figura (Subplot 2 ora usa smoothed_pose)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for ax, title in zip([ax1, ax2], ['Original', f'VQ-VAE']):
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
        ax.set_title(title)

    lines1 = [ax1.plot([], [], 'k-', linewidth=2, alpha=0.8)[0] for _ in range(len(parents))]
    scatter1 = ax1.scatter([], [], c='blue', s=20)
    lines2 = [ax2.plot([], [], 'r-', linewidth=2, alpha=0.8)[0] for _ in range(len(parents))]
    scatter2 = ax2.scatter([], [], c='darkred', s=20)

    def update(frame_idx):
        f1 = normalized_pose[frame_idx]
        f2 = smoothed_pose[frame_idx] # Usiamo la versione interpolata
        for i, p_idx in enumerate(parents):
            if p_idx >= 0:
                lines1[i].set_data([f1[i,0], f1[p_idx,0]], [f1[i,1], f1[p_idx,1]])
                lines2[i].set_data([f2[i,0], f2[p_idx,0]], [f2[i,1], f2[p_idx,1]])
        scatter1.set_offsets(f1); scatter2.set_offsets(f2)
        return lines1 + lines2 + [scatter1, scatter2]

    anim = FuncAnimation(fig, update, frames=len(normalized_pose), blit=True, interval=interval)
    plt.close()
    
    if output_filename:
        fps = int(1000 / interval)
        anim.save(output_filename, writer='ffmpeg', fps=fps, extra_args=['-vcodec', 'libx264'])
    
        print("Salvataggio completato.")
    print("Fine.")

# Esecuzione (passa il tuo oggetto modello caricato)
trials = [2, 8, 10, 26, 37, 41]
for trial in trials:
    model : SkeletonVQVAE = SkeletonVQVAE.load(f'weights/vqvae_trial_{trial}.pt', device='cpu')
    compare_quantized_animation(model, speaker="fallon", clip_idx=30, output_filename=f"videos/trial_{trial}_skeleton_comparison.mp4")