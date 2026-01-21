import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize


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
    def load(path: str, device='cpu'):
        """Carica il modello ricostruendo l'architettura dai metadati salvati."""
        checkpoint = torch.load(path, map_location=device)
        
        # Inizializza una nuova istanza con la configurazione salvata
        model = SkeletonVQVAE(**checkpoint['config'])
        
        # Carica i pesi
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval() # Di default in modalità valutazione
        return model