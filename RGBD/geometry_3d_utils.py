"""
geometry_3d_utils.py
3D/depth geometry operations: backprojection, pointcloud conversion,
in-plane pose composition, and rotation error. For 2D image/patch
operations, see image_patch_utils.py.
"""

import numpy as np


def add_inplane_to_pose(base_pose, inplane_rot_deg):
    """Composes an extra in-plane (camera-Z-axis) rotation onto a pose."""
    theta = np.radians(inplane_rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    R_inplane = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R_combined = R_inplane @ base_pose[:3, :3]
    rotated_pose = np.eye(4)
    rotated_pose[:3, :3] = R_combined
    rotated_pose[:3, 3] = base_pose[:3, 3]
    return rotated_pose

def depth_to_pointcloud(depth, K):
    """Backprojects a dense depth map to a (H, W, 3) pointcloud (per-pixel)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    points = np.stack((x, y, z), axis=-1)
    return points

def backproject_depth_to_3d(K, depth):
    """Backprojects only the nonzero-depth pixels; returns (pts, row_idx, col_idx)."""
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts, vs, us

def rotate_pointcloud(pc, angle_degree):
    """Rotates a pointcloud about the Z axis by angle_degree."""
    angle = np.deg2rad(angle_degree)
    Rz = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    h, w, _ = pc.shape
    pc_rot = pc @ Rz.T
    return pc_rot

def pointcloud_to_depth(pc_rotated, K):
    """Reprojects a pointcloud back to a depth map, keeping the nearest point per pixel."""
    x = pc_rotated[..., 0]
    y = pc_rotated[..., 1]
    z = pc_rotated[..., 2]
    h, w = z.shape

    u = (K[0, 0] * x / z + K[0, 2])
    v = (K[1, 1] * y / z + K[1, 2])

    depth_new = np.zeros((h, w), dtype=np.float32)

    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)

    valid = (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)

    for i in range(h):
        for j in range(w):
            if valid[i, j] and z[i, j] > 0:
                ui, vi = u_int[i, j], v_int[i, j]
                if depth_new[vi, ui] == 0 or z[i, j] < depth_new[vi, ui]:
                    depth_new[vi, ui] = z[i, j]

    return depth_new

def rotation_error(T1, T2):
    """Angular distance in degrees between two 3x3 rotations or 4x4 poses."""
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    cos_angle = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    # Clamp for numerical precision.
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    theta = np.arccos(cos_angle)

    return np.degrees(theta)
