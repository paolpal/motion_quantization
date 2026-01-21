import os
import json
import torch
import torch.nn as nn
import numpy as np
import optuna
import wandb
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize
from pats.utils import Skeleton2D

# --- CONFIGURAZIONE ---
WANDB_API_KEY = "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"
# DATA_ROOT = Path('/home/ubuntu/palumbo/Posemi/dataset/data/pats/data')
DATA_ROOT = Path('/home/paolo/Projects/Gesture/pats/data')
CACHE_DIR = Path('cache-dataset')
CACHE_DIR.mkdir(exist_ok=True)

STUDY_NAME = "local_test"
DB_NAME = f"sqlite:///{STUDY_NAME}.db"

# --- METRICA DI STABILITÀ (Jittering) ---
def calculate_stability_metric(y_true, y_pred):
    # y_true/pred shape: [T, 52, 2]
    mse_pos = torch.mean((y_pred - y_true)**2)
    
    # Velocità e Accelerazione
    vel_true = y_true[1:] - y_true[:-1]
    vel_pred = y_pred[1:] - y_pred[:-1]
    mse_vel = torch.mean((vel_pred - vel_true)**2)
    
    acc_true = vel_true[1:] - vel_true[:-1]
    acc_pred = vel_pred[1:] - vel_pred[:-1]
    mse_acc = torch.mean((acc_pred - acc_true)**2)
    
    eps = 1e-6
    # Usiamo la varianza dei dati reali come normalizzatore
    var_pos = torch.var(y_true) + eps
    var_vel = torch.var(vel_true) + eps
    var_acc = torch.var(acc_true) + eps
    
    return (mse_pos / var_pos) + (mse_vel / var_vel) + (mse_acc / var_acc)

# --- DATASET MULTI-SPEAKER CLIP-AWARE ---
class MultiSpeakerDataset(Dataset):
    def __init__(self, speakers, split="train", force_rebuild=False):
        self.speakers = sorted(speakers)
        spk_hash = "_".join(self.speakers[:3]) + f"_etc_{len(self.speakers)}"
        self.cache_file = CACHE_DIR / f"{spk_hash}_{split}.pt"
        
        if force_rebuild or not self.cache_file.exists():
            self._build_and_save(split)
        else:
            print(f"Caricamento cache: {self.cache_file}")
            data_dict = torch.load(self.cache_file, weights_only=False)
            self.clips = data_dict['clips']
            self.clip_indices = data_dict['clip_indices']
            self.all_frames = torch.cat(self.clips, dim=0)

    def _build_and_save(self, split):
        print(f"Generazione cache pesante per {len(self.speakers)} speaker...")
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

        torch.save({'clips': self.clips, 'clip_indices': self.clip_indices}, self.cache_file)

    def _load_speaker_clips(self, speaker, split):
        from pats.utils import load_multiple_samples, get_speaker_intervals
        intervals = get_speaker_intervals(speaker=speaker, split=split, data_root=DATA_ROOT)
        samples = load_multiple_samples(speaker=speaker, interval_ids=intervals, data_root=DATA_ROOT)
        processed = []
        for s in samples:
            p = s['pose']
            p[:, 0] = [0.0, 0.0] # Centering
            processed.append(p)
        return processed

    def __len__(self): return len(self.all_frames)
    def __getitem__(self, idx): return self.all_frames[idx]

# --- MODELLO VQ-VAE ---

class SkeletonVQVAE(nn.Module):
    def __init__(
        self,
        encoder_dims=[104, 128, 64, 32],
        decoder_dims=[32, 64, 128, 104],
        num_codes=64,
        dropout=0.2,
        commitment_weight=0.1,
        decay=0.95,
        threshold_ema_dead_code=10
    ):
        super().__init__()

        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.num_codes = num_codes
        self.dropout = dropout
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # ------- ENCODER -------
        enc_layers = []
        for i in range(len(encoder_dims) - 1):
            # Costruiamo i componenti base del blocco
            block = [
                nn.Linear(encoder_dims[i], encoder_dims[i+1]),
                nn.LayerNorm(encoder_dims[i+1])
            ]
            
            # Aggiungiamo ReLU e Dropout solo se non siamo nell'ultimo layer
            if i < len(encoder_dims) - 2:
                block.extend([
                    nn.ReLU(), 
                    nn.Dropout(dropout)
                ])
            
            enc_layers.append(nn.Sequential(*block))
            
        self.encoder = nn.Sequential(*enc_layers)

        latent_dim = encoder_dims[-1]

        # ------- VQ -------
        self.vq = VectorQuantize(
            dim = latent_dim,
            codebook_size = num_codes,
            decay = decay,
            commitment_weight = commitment_weight,
            kmeans_init = True,
            threshold_ema_dead_code = threshold_ema_dead_code,
            orthogonal_reg_weight=1,
            orthogonal_reg_active_codes_only=False,
            orthogonal_reg_max_codes=num_codes
        )

        # ------- DECODER -------
        dec_layers = []
        for i in range(len(decoder_dims) - 1):
            # Componente base: la trasformazione lineare
            block : list[nn.Module] = [
                nn.Linear(decoder_dims[i], decoder_dims[i+1])
            ]
            
            # Aggiungiamo l'attivazione solo se non siamo all'ultimo layer
            if i < len(decoder_dims) - 2:
                block.extend([
                    nn.LayerNorm(decoder_dims[i+1]),
                    nn.GELU()
                ]) 
            
            dec_layers.append(nn.Sequential(*block))

        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        # x: (B, 52, 2)
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        z_q, indices, vq_loss = self.vq(z)
        x_recon = self.decoder(z_q)
        x_recon = x_recon.view(-1, 52, 2)

        return x_recon, vq_loss, indices
    
    def save(self, path):
        """Salva i pesi e la configurazione del modello."""
        payload = {
            'state_dict': self.state_dict(),
            'config': {
                'encoder_dims': self.encoder_dims,
                'decoder_dims': self.decoder_dims,
                'num_codes': self.num_codes,
                'dropout': self.dropout,
                'commitment_weight': self.commitment_weight,
                'decay': self.decay,
                'threshold_ema_dead_code': self.threshold_ema_dead_code
            }
        }
        torch.save(payload, path)
        print(f"Modello salvato in: {path}")

    @staticmethod
    def load(path, device='cpu'):
        """Carica il modello ricostruendo l'architettura dai metadati salvati."""
        checkpoint = torch.load(path, map_location=device)
        
        # Inizializza una nuova istanza con la configurazione salvata
        model = SkeletonVQVAE(**checkpoint['config'])
        
        # Carica i pesi
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval() # Di default in modalità valutazione
        return model


# --- OPTUNA OBJECTIVE ---
def objective(trial):
    config = {
        "init_lr": trial.suggest_float("lr", 1e-7, 1e-4, log=True),
        "dropout": trial.suggest_float("dropout", 0.05, 0.2),
        "commitment_weight": trial.suggest_float("commitment_weight", 0.1, 0.5),
        "num_codes": trial.suggest_categorical("num_codes", [32, 48, 64, 80, 96, 128]),
        "latent_dim": trial.suggest_categorical("latent_dim", [16, 32, 64]),
        "batch_size": trial.suggest_categorical("batch_size", [64]),
        "decay": trial.suggest_float("decay", 0.75, 0.99),
        "threshold_ema": trial.suggest_int("threshold_ema_dead_code", 20, 100, step=10)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gkf = GroupKFold(n_splits=5)
    fold_stabilities = []

    for fold_id, (train_idx, val_idx) in enumerate(gkf.split(DATASET.all_frames.numpy(), groups=DATASET.clip_indices)):
        trial_id = f"T{trial.number}"

        run = wandb.init(
            project=STUDY_NAME, 
            group=trial_id,            # Raggruppa i 5 fold sotto "T0", "T1", ecc.
            name=f"{trial_id}_fold_{fold_id}", # Nome specifico della run: "T0_fold_0"
            job_type="cross-val",      # Aiuta a filtrare se farai altri tipi di test
            config=config, 
            reinit=True,
            tags=[f"gpu_{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"] # Tag per sapere quale H100 ha lavorato
        )

        model = SkeletonVQVAE(
            encoder_dims=[104, 128, 64, config["latent_dim"]],
            decoder_dims=[config["latent_dim"], 64, 128, 104],
            num_codes=config["num_codes"],
            dropout=config["dropout"],
            commitment_weight=config["commitment_weight"],
            decay=config["decay"],
            threshold_ema_dead_code=config["threshold_ema"]
        ).to(device)

        train_loader = DataLoader(Subset(DATASET, train_idx), batch_size=config["batch_size"], num_workers=8, prefetch_factor=2, shuffle=True, pin_memory=True, drop_last=True)
        val_loader = DataLoader(Subset(DATASET, val_idx), batch_size=config["batch_size"], num_workers=8, prefetch_factor=2)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["init_lr"], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        recon_criterion = nn.MSELoss()
        # Training 
        
        # Early Stopping params
        patience = 15
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, 301):
            model.train()
            train_recon, train_vq, total_perp = 0.0, 0.0, 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad()
                recon, vq_l, idxs = model(batch)
                
                # Perplexity
                counts = torch.histc(idxs.float(), bins=config["num_codes"], min=0, max=config["num_codes"]-1)
                probs = counts / (counts.sum() + 1e-10)
                perp = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
                total_perp += perp.item()
                
                r_loss = recon_criterion(recon, batch)
                loss = r_loss + vq_l
                loss.backward()
                optimizer.step()
                
                train_recon += r_loss.item() * batch.size(0)
                train_vq += vq_l.item() * batch.size(0)

            # Validation
            model.eval()
            val_recon, val_vq = 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device, non_blocking=True)
                    recon, vq_l, _ = model(batch)
                    val_recon += recon_criterion(recon, batch).item() * batch.size(0)
                    val_vq += vq_l.item() * batch.size(0)

            # CALCOLO MEDIE RICHIESTE
            avg_train_recon = train_recon / len(train_idx)
            avg_train_vq = train_vq / len(train_idx)
            avg_train = avg_train_recon + avg_train_vq
            
            avg_val_recon = val_recon / len(val_idx)
            avg_val_vq = val_vq / len(val_idx)
            avg_val = avg_val_recon + avg_val_vq
            
            avg_train_perplexity = total_perp / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']

            # LOG WANDB CON LE TUE CHIAVI ESATTE
            run.log({
                'train_loss': avg_train,
                'val_loss': avg_val,
                'lr': current_lr,
                'train_recon_loss': avg_train_recon,
                'train_vq_loss': avg_train_vq,
                'train_perplexity': avg_train_perplexity,
            })

            # Optuna Pruning
            if fold_id == 0:
                trial.report(avg_val, epoch)
            if trial.should_prune():
                run.finish(exit_code=1)
                raise optuna.exceptions.TrialPruned()

            # Early Stopping
            scheduler.step(avg_val)
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"Trial {trial.number} stopped at epoch {epoch}")
                break

        # Validazione Stabilità su CLIP INTERE
        model.eval()
        val_clip_ids = np.unique(DATASET.clip_indices[val_idx])
        stability_accum = 0.0
        
        with torch.no_grad():
            for c_id in val_clip_ids:
                clip = DATASET.clips[c_id].to(device)
                recon_clip, _, _ = model(clip)
                stability_accum += calculate_stability_metric(clip, recon_clip).item()
        
        avg_stability = stability_accum / len(val_clip_ids)
        fold_stabilities.append(avg_stability)
        
        wandb.log({"fold_stability": avg_stability})
        run.finish()

        model.save(CACHE_DIR / f"model_trial{trial.number}_fold{fold_id}.pt")

    return float(np.mean(fold_stabilities))

if __name__ == "__main__":
    speakers = ["almaram", "bee", "colbert", "corden", "fallon", "jon", "minhaj", "oliver", "seth",
                "angelica", "chemistry", "conan", "ellen", "huckabee", "maher", "noah", "rock", "shelly"]
    speakers = ["fallon"]

    # Inizializza dataset (una volta sola, caricherà la cache se esiste)
    DATASET = MultiSpeakerDataset(speakers=speakers, split="dev")

    wandb.login(key=WANDB_API_KEY)

    study = optuna.create_study(
        study_name=STUDY_NAME, storage=DB_NAME, direction="minimize", load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )

    # Con 2 H100, puoi lanciare questo script in due terminali diversi.
    # Se vuoi il parallelismo interno allo script, usa n_jobs=2 o superiore.
    study.optimize(objective, n_trials=10, n_jobs=1)