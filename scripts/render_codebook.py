import torch
import matplotlib.pyplot as plt
import numpy as np
from motion_quantization.models import SkeletonVQVAE
from motion_quantization.utils.skeletonPATS import SkeletonPATS

def plot_decoded_codebook_skeletons(model, num_rows=4, num_cols=8, figsize=(15, 8), save_path=None, dpi=300):
    """
    Decodifica una selezione di elementi dal codebook del VQ-VAE e li visualizza come scheletri.
    Se il codebook contiene meno elementi delle celle nella griglia, le celle rimanenti vengono lasciate vuote.
    """
    model.eval()
    parents = SkeletonPATS.parents()  # Assicurati che SkeletonPATS sia importato
    
    with torch.no_grad():
        # Recupero vettori dal codebook
        if hasattr(model.vq, 'codebook'):
            codebook_embeddings = model.vq.codebook
        else:
            codebook_embeddings = model.vq.embedding.weight 
        
        # Decodifica
        decoded_latent = model.decoder(codebook_embeddings)
        decoded_skeletons = decoded_latent.view(-1, 52, 2).cpu().numpy()

    # Numero totale di slot nella griglia
    total_slots = num_rows * num_cols
    num_codes = len(decoded_skeletons)

    # Indici sicuri da plottare
    indices_to_plot = np.arange(min(total_slots, num_codes))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle('Codebook VQ-VAE', fontsize=16, fontweight='bold')

    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes):
        if i < len(indices_to_plot):
            idx = indices_to_plot[i]
            frame = decoded_skeletons[idx]

            # Disegno ossa
            for j, parent_idx in enumerate(parents):
                if parent_idx >= 0:
                    ax.plot(
                        [frame[j, 0], frame[parent_idx, 0]],
                        [frame[j, 1], frame[parent_idx, 1]],
                        'k-', linewidth=1, alpha=0.8
                    )
            
            # Disegno giunti
            ax.scatter(frame[:, 0], frame[:, 1], c='red', s=10, alpha=0.6, zorder=5)

            ax.set_title(f"Codice #{idx}", fontsize=8)
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            ax.invert_yaxis()
        else:
            ax.axis('off')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Immagine salvata in {save_path}")

    # plt.show()


trials = [2, 8, 10, 26, 37, 41]
for trial in trials:
    model : SkeletonVQVAE = SkeletonVQVAE.load(f'weights/vqvae_trial_{trial}.pt', device='cpu')
    num_cols = 8
    num_rows = model.vq.codebook_size // num_cols + (1 if model.vq.codebook_size % num_cols != 0 else 0)
    plot_decoded_codebook_skeletons(model, num_rows=num_rows, num_cols=num_cols, figsize=(20, 2*num_rows), save_path=f'codebooks/trial_{trial}_codebook.png', dpi=400)