import torch
from typing import Optional, Tuple, Union
from motion_quantization.models import SkeletonVQVAE

class PoseTokenizer:
    """
    Interfaccia per l'inferenza di un modello SkeletonVQVAE.
    Gestisce la conversione tra pose (spazio continuo) e token (spazio discreto).
    """

    # --------
    # VERSIONE
    # --------
    VERSION = "1.0"

    # ----------------
    # TOKEN SPECIALI
    # ORDINE FISSO!
    # ----------------
    PAD = "<PAD>"
    SOM = "<SOM>" # Start Of Motion
    EOM = "<EOM>" # End Of Motion

    SPECIAL_TOKENS = [PAD, SOM, EOM]
    
    # Mapping token -> ID (PAD deve essere 0)
    TOKEN_TO_ID = {
        PAD: 0,
        SOM: 1,
        EOM: 2
    }
    
    # Mapping ID -> token
    ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_ID.items()}
    
    # Numero di token speciali (offset per gli indici del codebook)
    NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)

    def __init__(self, model: SkeletonVQVAE, device: Optional[Union[str, torch.device]] = None):
        # Imposta il device (usa quello del modello se non specificato)
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.eval()
        
        # Shortcut al codebook per lookup veloci
        self.codebook = model.vq.codebook
        
        # Dimensione totale del vocabolario (token speciali + codebook)
        self.vocab_size = self.NUM_SPECIAL_TOKENS + self.codebook.shape[0]
    
    # ----------------
    # PROPRIETÀ
    # ----------------
    @property
    def pad_id(self) -> int:
        """ID del token PAD (sempre 0)"""
        return self.TOKEN_TO_ID[self.PAD]
    
    @property
    def som_id(self) -> int:
        """ID del token SOM (Start Of Motion)"""
        return self.TOKEN_TO_ID[self.SOM]
    
    @property
    def eom_id(self) -> int:
        """ID del token EOM (End Of Motion)"""
        return self.TOKEN_TO_ID[self.EOM]
    
    @property
    def codebook_size(self) -> int:
        """Numero di vettori nel codebook (esclusi i token speciali)"""
        return self.codebook.shape[0]
    
    # ----------------
    # METODI UTILITÀ
    # ----------------
    def is_special_token(self, token_id: int) -> bool:
        """Verifica se un ID corrisponde a un token speciale"""
        return token_id < self.NUM_SPECIAL_TOKENS
    
    def codebook_idx_to_token_id(self, codebook_idx: torch.Tensor) -> torch.Tensor:
        """
        Converte gli indici del codebook (0-based) in ID token (offset di NUM_SPECIAL_TOKENS).
        Es: codebook_idx=0 -> token_id=3 (se ci sono 3 token speciali)
        """
        return codebook_idx + self.NUM_SPECIAL_TOKENS
    
    def token_id_to_codebook_idx(self, token_id: torch.Tensor) -> torch.Tensor:
        """
        Converte gli ID token in indici del codebook (rimuove l'offset).
        Attenzione: non verifica se token_id è un token speciale!
        """
        return token_id - self.NUM_SPECIAL_TOKENS

    @torch.no_grad()
    def encode(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Mappa la posa nello spazio latente continuo (z).
        Input: (B, T, J, C) -> Output: (B, D, T')
        """
        return self.model.encoder(pose.to(self.device))

    @torch.no_grad()
    def quantize(self, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mappa la posa ai vettori del codebook e ai relativi indici.
        Output: (z_q, token_ids)
        NOTA: gli indici restituiti sono già convertiti in token_ids (con offset).
        """
        z = self.encode(pose)
        # Il modulo VQ restituisce (quantized, loss, indices)
        z_q, codebook_indices, _ = self.model.vq(z)
        # Converti gli indici del codebook in token IDs
        token_ids = self.codebook_idx_to_token_id(codebook_indices)
        return z_q, token_ids

    @torch.no_grad()
    def dequantize(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Ricostruisce la posa partendo dai token IDs.
        Input: token_ids (B, T') -> Output: posa (B, T, J, C)
        NOTA: i token speciali non possono essere decodificati e causeranno errori.
        """
        z_q = self.lookup(token_ids)
        return self.model.decoder(z_q)

    @torch.no_grad()
    def lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Recupera i vettori continui dal codebook usando i token IDs.
        Converte automaticamente i token_ids in indici del codebook.
        NOTA: i token speciali (ID < NUM_SPECIAL_TOKENS) causeranno indici negativi!
        """
        codebook_indices = self.token_id_to_codebook_idx(token_ids)
        return self.codebook[codebook_indices]

    @torch.no_grad()
    def reconstruct(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Esegue il ciclo completo: encode -> quantize -> decode.
        Utile per valutare visivamente la perdita di qualità.
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
        Aggiunge i token speciali SOM e/o EOM a una sequenza di token IDs.
        
        Args:
            token_ids: Tensor di forma (B, T) con gli ID dei token
            add_som: Se True, aggiunge SOM all'inizio
            add_eom: Se True, aggiunge EOM alla fine
            
        Returns:
            Tensor di forma (B, T + num_added) con i token speciali
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
        Rimuove tutti i token speciali (PAD, SOM, EOM) da una sequenza.
        
        Args:
            token_ids: Tensor di forma (B, T) con gli ID dei token
            
        Returns:
            Lista di tensori, uno per ogni batch, senza token speciali
        """
        batch_size, seq_len = token_ids.shape
        result = []
        
        for b in range(batch_size):
            # Maschera per i token non speciali
            mask = token_ids[b] >= self.NUM_SPECIAL_TOKENS
            filtered = token_ids[b][mask]
            result.append(filtered)
        
        return result
    
    def pad_sequence(self, token_ids: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Applica padding a una sequenza di token fino a max_length.
        
        Args:
            token_ids: Tensor di forma (B, T) con gli ID dei token
            max_length: Lunghezza target
            
        Returns:
            Tensor di forma (B, max_length) con padding applicato
        """
        batch_size, seq_len = token_ids.shape
        
        if seq_len >= max_length:
            return token_ids[:, :max_length]
        
        pad_len = max_length - seq_len
        pad_tokens = torch.full((batch_size, pad_len), self.pad_id,
                               dtype=token_ids.dtype, device=token_ids.device)
        
        return torch.cat([token_ids, pad_tokens], dim=1)
