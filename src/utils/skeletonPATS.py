import numpy as np
from typing import Optional

class SkeletonPATS:
    def __init__(self):
        pass

    @staticmethod
    def parents() -> list[int]:
        return [-1,
            0, 1, 2,
            0, 4, 5,
            0, 7, 7,
            6,
            10, 11, 12, 13,
            10, 15, 16, 17,
            10, 19, 20, 21,
            10, 23, 24, 25,
            10, 27, 28, 29,
            3,
            31, 32, 33, 34,
            31, 36, 37, 38,
            31, 40, 41, 42,
            31, 44, 45, 46,
            31, 48, 49, 50]

    @staticmethod
    def joint_names() -> list[str]:
        return ['Neck',
                'RShoulder', 'RElbow', 'RWrist',
                'LShoulder', 'LElbow', 'LWrist',
                'Nose', 'REye', 'LEye',
                'LHandRoot',
                'LHandThumb1', 'LHandThumb2', 'LHandThumb3', 'LHandThumb4',
                'LHandIndex1', 'LHandIndex2', 'LHandIndex3', 'LHandIndex4',
                'LHandMiddle1', 'LHandMiddle2', 'LHandMiddle3', 'LHandMiddle4',
                'LHandRing1', 'LHandRing2', 'LHandRing3', 'LHandRing4',
                'LHandLittle1', 'LHandLittle2', 'LHandLittle3', 'LHandLittle4',
                'RHandRoot',
                'RHandThumb1', 'RHandThumb2', 'RHandThumb3', 'RHandThumb4',
                'RHandIndex1', 'RHandIndex2', 'RHandIndex3', 'RHandIndex4',
                'RHandMiddle1', 'RHandMiddle2', 'RHandMiddle3', 'RHandMiddle4',
                'RHandRing1', 'RHandRing2', 'RHandRing3', 'RHandRing4',
                'RHandLittle1', 'RHandLittle2', 'RHandLittle3', 'RHandLittle4'
        ]

    @staticmethod
    def filter_skeleton(
        pose: np.ndarray,
        remove_head: bool = True,
        remove_hands: bool = False,
    ) -> np.ndarray:
        """
        Supports:
          - (J, D)
          - (T, J, D)
          - (B, T, J, D)
        """
        names = SkeletonPATS.joint_names()
        J = len(names)

        if pose.ndim not in (2, 3, 4):
            raise ValueError(
                f"Unsupported pose shape {pose.shape}. "
                "Expected (J, D), (T, J, D) or (B, T, J, D)."
            )

        if pose.shape[-2] != J:
            raise ValueError(
                f"Joint dimension mismatch: expected {J}, got {pose.shape[-2]}"
            )

        remove: set[str] = set({'Neck'})

        if remove_head:
            remove.update({'Nose', 'REye', 'LEye'})

        if remove_hands:
            remove.update(name for name in names if 'Hand' in name)

        keep_idx = np.array(
            [i for i, name in enumerate(names) if name not in remove],
            dtype=np.int64
        )

        return np.take(pose, keep_idx, axis=-2)

    @staticmethod
    def normalize_skeleton(pose: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        Normalizza lo scheletro centrando e scalando le posizioni dei joint.
        
        Args:
            pose: Array di shape (J, 2), (T, J, 2), o (B, T, J, 2)
            scale: Fattore di scala per la normalizzazione
        
        Returns:
            Array di shape identica all'input con le posizioni normalizzate
        """
        # Indici delle spalle nel sistema PATS
        l_shoulder = 4  # LShoulder
        r_shoulder = 1  # RShoulder
        
        # Calcola la distanza tra le spalle
        # Gestisce diverse dimensioni
        if pose.ndim == 2:
            shoulder_dist = np.linalg.norm(pose[l_shoulder] - pose[r_shoulder])
        elif pose.ndim == 3:
            shoulder_dist = np.linalg.norm(pose[:, l_shoulder] - pose[:, r_shoulder], axis=-1, keepdims=True)
            shoulder_dist = np.expand_dims(shoulder_dist, axis=-1)  # (T, 1, 1)
        elif pose.ndim == 4:
            shoulder_dist = np.linalg.norm(pose[:, :, l_shoulder] - pose[:, :, r_shoulder], axis=-1, keepdims=True)
            shoulder_dist = np.expand_dims(shoulder_dist, axis=-1)  # (B, T, 1, 1)
        else:
            raise ValueError(
                f"Unsupported pose shape {pose.shape}. "
                "Expected (J, 2), (T, J, 2) or (B, T, J, 2)."
            )
        
        # Evita divisione per zero
        shoulder_dist = np.maximum(shoulder_dist, 1e-8)
        normalized = pose / shoulder_dist * scale
        
        return normalized
    
    @staticmethod
    def encode_kinematics(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parents = np.array(SkeletonPATS.parents())
        T, J, _ = pose.shape
        
        # Escludiamo il root (indice 0) dai dati polari
        polar = np.zeros((T, J - 1, 3))
        root_pos = pose[:, 0, :].copy() 

        for j in range(1, J): # Partiamo da 1
            p = parents[j]
            # Calcolo vettori e angoli (stessa logica precedente)
            vec_j = pose[:, j, :] - pose[:, p, :]
            dist = np.linalg.norm(vec_j, axis=-1)
            phi_j = np.arctan2(vec_j[:, 1], vec_j[:, 0])

            gp = parents[p]
            if gp == -1:
                theta_local = phi_j
            else:
                vec_p = pose[:, p, :] - pose[:, gp, :]
                phi_p = np.arctan2(vec_p[:, 1], vec_p[:, 0])
                theta_local = phi_j - phi_p

            # Riempiamo l'indice j-1 perché abbiamo rimosso il root
            polar[:, j-1, 0] = dist
            polar[:, j-1, 1] = np.cos(theta_local)
            polar[:, j-1, 2] = np.sin(theta_local)
            
        return polar, root_pos

    @staticmethod
    def decode_kinematics(polar: np.ndarray, root_pos: Optional[np.ndarray]=None) -> np.ndarray:
        parents = np.array(SkeletonPATS.parents())
        T, J_minus_1, _ = polar.shape
        J = J_minus_1 + 1
        
        pose = np.zeros((T, J, 2))
        if root_pos is not None:
            pose[:, 0, :] = root_pos
        global_angles = np.zeros((T, J))

        for j in range(1, J): # Ricostruiamo partendo dal primo figlio del root
            p = parents[j]
            
            # Recuperiamo i dati dall'indice j-1
            dist = polar[:, j-1, 0]
            theta_local = np.arctan2(polar[:, j-1, 2], polar[:, j-1, 1])
            
            if parents[p] == -1:
                global_angles[:, j] = theta_local
            else:
                global_angles[:, j] = global_angles[:, p] + theta_local
                
            pose[:, j, 0] = pose[:, p, 0] + dist * np.cos(global_angles[:, j])
            pose[:, j, 1] = pose[:, p, 1] + dist * np.sin(global_angles[:, j])
            
        return pose