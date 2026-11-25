"""SE(3) helpers. Poses are 4x4 homogeneous matrices T_world_sensor.

Conventions
-----------
* Quaternions are (qx, qy, qz, qw), the order used by ROS and the ColoRadar files.
* The relative pose between frames 1 and 2 is T_1_2 = inv(T_w_1) @ T_w_2, i.e. the
  pose of frame 2 expressed in frame 1. This is the only quantity observable from a
  pair of sensor-frame images; world-frame differences (p2 - p1) are not.
* A 6-vector "twist" label is [tx, ty, tz, rx, ry, rz] with r the rotation vector
  (axis * angle, radians) of R_1_2.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def pose_from_pq(p: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """(..., 3) positions and (..., 4) quaternions -> (..., 4, 4) poses."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q_xyzw, dtype=np.float64)
    shape = p.shape[:-1]
    T = np.zeros(shape + (4, 4))
    T[..., :3, :3] = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(shape + (3, 3))
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T


def inv(T: np.ndarray) -> np.ndarray:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Ti = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    Ti[..., 3, 3] = 1.0
    return Ti


def relative(T_w_1: np.ndarray, T_w_2: np.ndarray) -> np.ndarray:
    """Pose of frame 2 expressed in frame 1."""
    return inv(T_w_1) @ T_w_2


def to_twist(T: np.ndarray) -> np.ndarray:
    """(..., 4, 4) -> (..., 6) [t, rotvec]."""
    shape = T.shape[:-2]
    rv = Rotation.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_rotvec().reshape(shape + (3,))
    return np.concatenate([T[..., :3, 3], rv], axis=-1)


def from_twist(x: np.ndarray) -> np.ndarray:
    """(..., 6) [t, rotvec] -> (..., 4, 4)."""
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape[:-1]
    T = np.zeros(shape + (4, 4))
    T[..., :3, :3] = Rotation.from_rotvec(x[..., 3:].reshape(-1, 3)).as_matrix().reshape(shape + (3, 3))
    T[..., :3, 3] = x[..., :3]
    T[..., 3, 3] = 1.0
    return T


def compose_trajectory(rel: np.ndarray, T0: np.ndarray | None = None) -> np.ndarray:
    """Chain N relative poses T_k_k+1 into N+1 absolute poses (proper SE(3) composition)."""
    T = np.eye(4) if T0 is None else T0.copy()
    out = [T.copy()]
    for Tk in rel:
        T = T @ Tk
        out.append(T.copy())
    return np.stack(out)


def rotation_angle(R: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) of rotation matrices."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def interpolate_poses(t_query: np.ndarray, t_ref: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
    """Linear position + SLERP rotation interpolation. t_query must lie inside t_ref."""
    t_ref = np.asarray(t_ref, dtype=np.float64)
    if np.any(np.diff(t_ref) <= 0):
        raise ValueError("reference timestamps must be strictly increasing")
    if t_query.min() < t_ref[0] or t_query.max() > t_ref[-1]:
        raise ValueError("query timestamps outside reference range")
    p = np.stack([np.interp(t_query, t_ref, T_ref[:, i, 3]) for i in range(3)], axis=-1)
    slerp = Slerp(t_ref, Rotation.from_matrix(T_ref[:, :3, :3]))
    out = np.zeros((len(t_query), 4, 4))
    out[:, :3, :3] = slerp(t_query).as_matrix()
    out[:, :3, 3] = p
    out[:, 3, 3] = 1.0
    return out


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid (no scale) T minimizing ||dst - T*src||. src/dst are (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    return T
