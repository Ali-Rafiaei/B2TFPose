import numpy as np


def add_inplane_to_pose(base_pose, inplane_rot_deg):
    theta = np.radians(inplane_rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    R_inplane = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R_combined = R_inplane @ base_pose[:3, :3]
    rotated_pose = np.eye(4)
    rotated_pose[:3, :3] = R_combined
    rotated_pose[:3, 3] = base_pose[:3, 3]
    return rotated_pose

def rotation_error(T1, T2):
    """
    Compute the rotation error between two 3x3 rotation or 4x4 transformation matrices.
    T1, T2: 3x3 or 4x4 numpy arrays
    """
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    cos_angle = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to handle numerical precision issues
    theta = np.arccos(cos_angle)

    return np.degrees(theta)
