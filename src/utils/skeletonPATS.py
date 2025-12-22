import numpy as np

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
    def compute_bone_lengths(pose: np.ndarray) -> np.ndarray:
        """
        Calcola le lunghezze delle ossa (distanze tra joint connessi).
        
        Args:
            pose: Array di shape (J, 2), (T, J, 2), o (B, T, J, 2)
                  dove J è il numero di joint
        
        Returns:
            Array di shape (..., J) con le distanze dal parent per ogni joint.
            Il primo joint (root) avrà distanza 0.
        """
        parents = np.array(SkeletonPATS.parents())
        
        # Gestisce diverse dimensioni
        original_shape = pose.shape
        if pose.ndim == 2:
            pose = pose[None, None, :, :]  # (1, 1, J, 2)
        elif pose.ndim == 3:
            pose = pose[None, :, :, :]  # (1, T, J, 2)
        
        B, T, J, D = pose.shape
        
        # Calcola le distanze
        distances = np.zeros((B, T, J))
        
        for j in range(J):
            parent_idx = parents[j]
            if parent_idx >= 0:
                # Distanza euclidea tra joint e parent
                diff = pose[:, :, j, :] - pose[:, :, parent_idx, :]
                distances[:, :, j] = np.linalg.norm(diff, axis=-1)
        
        # Ripristina la shape originale
        if len(original_shape) == 2:
            distances = distances[0, 0, :]
        elif len(original_shape) == 3:
            distances = distances[0, :, :]
        
        return distances

    @staticmethod
    def compute_bone_angles(pose: np.ndarray) -> np.ndarray:
        """
        Calcola gli angoli delle ossa rispetto al parent.
        Per ogni joint, calcola l'angolo del vettore (parent -> joint) 
        rispetto all'asse x.
        
        Args:
            pose: Array di shape (J, 2), (T, J, 2), o (B, T, J, 2)
                  dove J è il numero di joint
        
        Returns:
            Array di shape (..., J) con gli angoli in radianti.
        """
        parents = np.array(SkeletonPATS.parents())
        
        # Gestisce diverse dimensioni
        original_shape = pose.shape
        if pose.ndim == 2:
            pose = pose[None, None, :, :]  # (1, 1, J, 2)
        elif pose.ndim == 3:
            pose = pose[None, :, :, :]  # (1, T, J, 2)
        
        B, T, J, D = pose.shape
        
        # Angoli 2D
        angles = np.zeros((B, T, J))
        
        for j in range(J):
            parent_idx = parents[j]
            if parent_idx >= 0:
                # Vettore dal parent al joint
                diff = pose[:, :, j, :] - pose[:, :, parent_idx, :]
                # Angolo rispetto all'asse x
                angles[:, :, j] = np.arctan2(diff[:, :, 1], diff[:, :, 0])
        
        # Ripristina la shape originale
        if len(original_shape) == 2:
            angles = angles[0, 0, :]
        elif len(original_shape) == 3:
            angles = angles[0, :, :]
        
        return angles

    @staticmethod
    def encode_as_polar(pose: np.ndarray) -> np.ndarray:
        """
        Encodifica lo scheletro come angoli e distanze (coordinate polari).
        Ogni joint viene rappresentato come (distanza, angolo) invece di (x, y).
        
        Args:
            pose: Array di shape (J, 2), (T, J, 2), o (B, T, J, 2)
        
        Returns:
            Array di shape identica all'input dove ogni joint ha formato (r, θ):
            - r: distanza dal parent
            - θ: angolo rispetto all'asse x (in radianti)
        """
        distances = SkeletonPATS.compute_bone_lengths(pose)  # (..., J)
        angles = SkeletonPATS.compute_bone_angles(pose)  # (..., J)
        
        # Combina in un unico array mantenendo la shape originale
        # Espandi le dimensioni per lo stack
        # Funziona per tutte le dimensioni supportate
        polar = np.stack([distances, angles], axis=-1)
        return polar

    @staticmethod
    def decode_from_polar(
        polar: np.ndarray,
        root_position: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Decodifica lo scheletro da coordinate polari a posizioni cartesiane.
        
        Args:
            polar: Array di shape (J, 2), (T, J, 2), o (B, T, J, 2)
                  dove ogni joint ha formato (r, θ)
            root_position: Posizione del root joint. Se None, usa l'origine.
                          Shape: (2,), (T, 2), o (B, T, 2)
        
        Returns:
            Array di shape identica all'input con le posizioni (x, y) dei joint
        """
        parents = np.array(SkeletonPATS.parents())
        
        # Estrai distanze e angoli
        distances = polar[..., 0]  # (..., J)
        angles = polar[..., 1]  # (..., J)
        
        # Gestisce diverse dimensioni
        original_shape = polar.shape
        if polar.ndim == 2:
            distances = distances[None, None, :]  # (1, 1, J)
            angles = angles[None, None, :]  # (1, 1, J)
        elif polar.ndim == 3:
            distances = distances[None, :, :]  # (1, T, J)
            angles = angles[None, :, :]  # (1, T, J)
        
        B, T, J = distances.shape
        
        pose = np.zeros((B, T, J, 2))
        
        # Imposta la posizione del root
        if root_position is not None:
            if root_position.ndim == 1:
                root_position = root_position[None, None, :]  # (1, 1, 2)
            elif root_position.ndim == 2:
                root_position = root_position[None, :, :]  # (1, T, 2)
            pose[:, :, 0, :] = root_position
        
        # Ricostruisce le posizioni iterando sui joint
        for j in range(1, J):
            parent_idx = parents[j]
            if parent_idx >= 0:
                parent_pos = pose[:, :, parent_idx, :]
                dist = distances[:, :, j]
                angle = angles[:, :, j]
                
                # Converti da polare a cartesiano
                dx = dist * np.cos(angle)
                dy = dist * np.sin(angle)
                pose[:, :, j, :] = parent_pos + np.stack([dx, dy], axis=-1)
        
        # Ripristina la shape originale
        if len(original_shape) == 2:
            pose = pose[0, 0, :, :]
        elif len(original_shape) == 3:
            pose = pose[0, :, :, :]
        
        return pose 
    

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