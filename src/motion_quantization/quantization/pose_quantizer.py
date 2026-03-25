import torch
from typing import Optional, Tuple, Union
from motion_quantization.models import SkeletonVQVAE

class PoseTokenizer:
    """
    Interfaccia per l'inferenza di un modello SkeletonVQVAE.
    I token speciali (PAD, SOM, EOM) sono posizionati DOPO il codebook.
    """

    # --------
    # VERSIONE
    # --------
    VERSION = "1.1-special-at-end"

    def __init__(self, model: SkeletonVQVAE, device: Optional[Union[str, torch.device]] = None):
        # Imposta il device (usa quello del modello se non specificato)
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.eval()
        
        # Shortcut al codebook
        self.codebook = model.vq.codebook
        self.num_embeddings = self.codebook.shape[0]
        
        # ----------------
        # TOKEN SPECIALI (In fondo)
        # ----------------
        self.PAD = "<PAD>"
        self.SOM = "<SOM>"
        self.EOM = "<EOM>"
        
        self.SPECIAL_TOKENS = [self.PAD, self.SOM, self.EOM]
        
        # Mapping token -> ID (Offset basato sulla dimensione del codebook)
        self.TOKEN_TO_ID = {
            self.PAD: self.num_embeddings,
            self.SOM: self.num_embeddings + 1,
            self.EOM: self.num_embeddings + 2
        }
        
        # Mapping ID -> token
        self.ID_TO_TOKEN = {v: k for k, v in self.TOKEN_TO_ID.items()}
        # Aggiungiamo anche i nomi dei token del codebook per completezza (opzionale)
        for i in range(self.num_embeddings):
            self.ID_TO_TOKEN[i] = f"CB_{i}"
        
        # Dimensione totale del vocabolario
        self.vocab_size = self.num_embeddings + len(self.SPECIAL_TOKENS)

    # ----------------
    # PROPRIETÀ
    # ----------------
    @property
    def pad_id(self) -> int:
        """ID del token PAD (ora è N)"""
        return self.TOKEN_TO_ID[self.PAD]
    
    @property
    def som_id(self) -> int:
        """ID del token SOM (N + 1)"""
        return self.TOKEN_TO_ID[self.SOM]
    
    @property
    def eom_id(self) -> int:
        """ID del token EOM (N + 2)"""
        return self.TOKEN_TO_ID[self.EOM]
    
    @property
    def codebook_size(self) -> int:
        """Numero di vettori nel codebook"""
        return self.num_embeddings
    
    # ----------------
    # METODI UTILITÀ
    # ----------------
    def is_special_token(self, token_id: int) -> bool:
        """Verifica se un ID corrisponde a un token speciale (ID >= N)"""
        return token_id >= self.num_embeddings
    
    def codebook_idx_to_token_id(self, codebook_idx: torch.Tensor) -> torch.Tensor:
        """
        Converte gli indici del codebook in ID token.
        Essendo gli speciali in fondo, l'indice coincide con l'ID.
        """
        return codebook_idx.long()
    
    def token_id_to_codebook_idx(self, token_id: torch.Tensor) -> torch.Tensor:
        """
        Converte gli ID token in indici del codebook.
        Ritorna l'ID così com'è. 
        ATTENZIONE: Se passi un token speciale, causerai un Index Error nel codebook.
        """
        return token_id.long()

    @torch.no_grad()
    def encode(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Mappa la posa nello spazio latente continuo (z).
        """
        return self.model.encoder(pose.to(self.device))

    @torch.no_grad()
    def quantize(self, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mappa la posa ai vettori del codebook e ai relativi indici.
        """
        z = self.encode(pose)
        z_q, codebook_indices, _ = self.model.vq(z)
        token_ids = self.codebook_idx_to_token_id(codebook_indices)
        return z_q, token_ids

    @torch.no_grad()
    def dequantize(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Ricostruisce la posa partendo dai token IDs.
        """
        z_q = self.lookup(token_ids)
        return self.model.decoder(z_q)

    @torch.no_grad()
    def lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Recupera i vettori continui dal codebook usando i token IDs.
        """
        # Verifichiamo che non ci siano token speciali per evitare crash
        if (token_ids >= self.num_embeddings).any():
            raise ValueError("Impossibile eseguire il lookup di token speciali (PAD/SOM/EOM) nel codebook.")
            
        codebook_indices = self.token_id_to_codebook_idx(token_ids)
        return self.codebook[codebook_indices]

    @torch.no_grad()
    def reconstruct(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Esegue il ciclo completo: encode -> quantize -> decode.
        """
        z_q, _ = self.quantize(pose)
        return self.model.decoder(z_q)
    
    # --------------------------------
    # METODI PER GESTIRE SEQUENZE
    # --------------------------------
    def add_special_tokens(self, token_ids: torch.Tensor, 
                          add_som: bool = True, 
                          add_eom: bool = True) -> torch.Tensor:
        """
        Aggiunge i token speciali SOM e/o EOM a una sequenza.
        """
        batch_size = token_ids.shape[0]
        device = token_ids.device
        
        tokens = [token_ids]
        
        if add_som:
            som_tokens = torch.full((batch_size, 1), self.som_id, 
                                   dtype=token_ids.dtype, device=device)
            tokens.insert(0, som_tokens)
        
        if add_eom:
            eom_tokens = torch.full((batch_size, 1), self.eom_id, 
                                   dtype=token_ids.dtype, device=device)
            tokens.append(eom_tokens)
        
        return torch.cat(tokens, dim=1)
    
    def remove_special_tokens(self, token_ids: torch.Tensor) -> list:
        """
        Rimuove tutti i token speciali (ID >= num_embeddings) da una sequenza.
        """
        batch_size = token_ids.shape[0]
        result = []
        
        for b in range(batch_size):
            # Maschera per i token che appartengono al codebook
            mask = token_ids[b] < self.num_embeddings
            filtered = token_ids[b][mask]
            result.append(filtered)
        
        return result
    
    def pad_sequence(self, token_ids: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Applica padding a una sequenza di token fino a max_length.
        """
        batch_size, seq_len = token_ids.shape
        
        if seq_len >= max_length:
            return token_ids[:, :max_length]
        
        pad_len = max_length - seq_len
        pad_tokens = torch.full((batch_size, pad_len), self.pad_id,
                               dtype=token_ids.dtype, device=token_ids.device)
        
        return torch.cat([token_ids, pad_tokens], dim=1)