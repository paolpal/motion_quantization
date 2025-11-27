# YOLO Pose connections (17 keypoints)
KEYPOINTS = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

# Reverse lookup (name -> index)
KEYPOINT_INDEX = {name: idx for idx, name in KEYPOINTS.items()}

# Full skeleton connections (17 keypoints)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders
    (5, 6),          # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 11), (6, 12),# shoulders to hips
    (11, 12),        # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16)  # right leg
]

# Torso connections (no legs, 13 keypoints: 0-12)
TORSO_CONNECTIONS = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders
    (5, 6),          # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 11), (6, 12),# shoulders to hips
    (11, 12),        # hips
]

# Upper body connections (no legs, no head, 8 keypoints)
# Mapping: shoulders(0,1), elbows(2,3), wrists(4,5), hips(6,7)
UPPER_BODY_CONNECTIONS = [
    (0, 1),          # shoulders
    (0, 2), (2, 4),  # left arm
    (1, 3), (3, 5),  # right arm
    (0, 6), (1, 7),  # shoulders to hips
    (6, 7),          # hips
]