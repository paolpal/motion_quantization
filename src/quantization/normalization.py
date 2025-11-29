from typing import Optional
import numpy as np
from utils.plot import plot_skeleton
from utils.constants import KEYPOINT_INDEX

def normalize_pose(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalizza i keypoints rispetto al corpo (centro e scala).
    keypoints : np.ndarray
        Shape (num_keypoints, 2)
    Returns: keypoints normalizzati come np.ndarray flattenato
    """
    kp = keypoints    
    
    center = kp.mean(axis=0)
    
    # Scala: distanza media dal centro
    distances = np.linalg.norm(kp - center, axis=0)
    scale = distances.mean()
    
    if scale < 1e-6:
        normalized= kp - center
    else:
        normalized = (kp - center) / scale

    return normalized

def filter_torso(keypoints: np.ndarray, remove_head=True) -> np.ndarray:
    """
    Filtra i keypoints per mantenere solo torso e braccia, rimuovendo gambe e opzionalmente la testa.
    Mantiene: spalle, gomiti, polsi, anche (e opzionalmente testa)
    Rimuove sempre: ginocchia, caviglie
    Rimuove opzionalmente: naso, occhi, orecchie
    
    keypoints: lista di [x, y] per ogni punto (17 punti YOLO)
    remove_head: se True rimuove naso, occhi e orecchie (default: True)
    Returns: keypoints filtrati come lista (stessa lunghezza, punti rimossi = [0, 0])
    """
    
    # Indici dei keypoints da mantenere (YOLO Pose)
    # Spalle (5,6), gomiti (7,8), polsi (9,10), anche (11,12)
    keep_indices = np.array([
        KEYPOINT_INDEX['left_shoulder'],
        KEYPOINT_INDEX['right_shoulder'],
        KEYPOINT_INDEX['left_elbow'],
        KEYPOINT_INDEX['right_elbow'],
        KEYPOINT_INDEX['left_wrist'],
        KEYPOINT_INDEX['right_wrist'],
        KEYPOINT_INDEX['left_hip'],
        KEYPOINT_INDEX['right_hip']
    ])
    
    # Se non si rimuove la testa, aggiungere i punti della testa
    if not remove_head:
        head_indices = np.array([
            KEYPOINT_INDEX['nose'],
            KEYPOINT_INDEX['left_eye'],
            KEYPOINT_INDEX['right_eye'],
            KEYPOINT_INDEX['left_ear'],
            KEYPOINT_INDEX['right_ear']
        ])
        keep_indices = np.concatenate([keep_indices, head_indices])

    return keypoints[keep_indices]