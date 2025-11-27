import numpy as np
from utils.plot import plot_skeleton
from utils.constants import KEYPOINT_INDEX

def normalize_pose(keypoints):
    """
    Normalizza i keypoints rispetto al corpo (centro e scala).
    keypoints: lista di [x, y] per ogni punto
    Returns: keypoints normalizzati come array flat
    """
    if keypoints is None:
        return None
    
    kp = np.array(keypoints)
    
    if kp.size == 0:
        return None
    
    center = kp.mean(axis=0)
    
    # Scala: distanza media dal centro
    distances = np.linalg.norm(kp - center, axis=0)
    scale = distances.mean()
    
    if scale < 1e-6:
        return None
    
    # Normalizza
    normalized = (kp - center) / scale

    return normalized.flatten()

def filter_torso(keypoints, remove_head=True):
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
    keep_indices = [
        KEYPOINT_INDEX['left_shoulder'],
        KEYPOINT_INDEX['right_shoulder'],
        KEYPOINT_INDEX['left_elbow'],
        KEYPOINT_INDEX['right_elbow'],
        KEYPOINT_INDEX['left_wrist'],
        KEYPOINT_INDEX['right_wrist'],
        KEYPOINT_INDEX['left_hip'],
        KEYPOINT_INDEX['right_hip']
    ]
    
    # Se non si rimuove la testa, aggiungere i punti della testa
    if not remove_head:
        keep_indices.extend([
            KEYPOINT_INDEX['nose'],
            KEYPOINT_INDEX['left_eye'],
            KEYPOINT_INDEX['right_eye'],
            KEYPOINT_INDEX['left_ear'],
            KEYPOINT_INDEX['right_ear']
        ])

    if keypoints is None or len(keypoints) == 0:
        return keypoints
    filtered_keypoints = [keypoints[i] for i in keep_indices]
    return filtered_keypoints