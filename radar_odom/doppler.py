"""Doppler ego-velocity: classical least squares and a learned robust variant.

Physics. For a static reflector at p = r*u in the radar frame, with radar linear velocity
v and angular velocity w, the measured radial velocity is

    d = u . (-(v + w x p)) = -u . v          (u . (w x p) = 0 since p is parallel to u)

so a single frame determines v from the Doppler of static returns, and carries NO
information about rotation. Moving objects, multipath and noise violate the model.

DopplerVelocityNet (proposed): a PointNet scores each return (trust weight), and v is
the closed-form weighted least-squares solution, differentiable end to end. Two rounds:
the second round sees each point's residual under the first estimate (a learned IRLS).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import PolarGridCfg
from .geometry import relative

FEATS = ["ux", "uy", "uz", "range", "doppler", "intensity"]


def unit_vectors(grid: PolarGridCfg) -> np.ndarray:
    E, A = np.meshgrid(grid.elevations(), grid.azimuths(), indexing="ij")
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)  # (E, A, 3)


def extract_points(stored: np.ndarray, grid: PolarGridCfg, k: int = 512) -> np.ndarray:
    """Strongest k cells of one stored frame (C=2, E, A, R; log1p intensity) -> (k, 6)."""
    inten = stored[0].astype(np.float32)
    inten = inten - np.median(inten)
    flat = inten.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    e, a, r = np.unravel_index(idx, inten.shape)
    u = unit_vectors(grid)[e, a]
    rng = grid.ranges()[r]
    dop = stored[1].astype(np.float32)[e, a, r]
    w = flat[idx] / max(float(flat[idx].max()), 1e-6)
    return np.concatenate([u, rng[:, None], dop[:, None], w[:, None]], 1).astype(np.float32)


def velocity_labels(poses: np.ndarray, stamps: np.ndarray) -> np.ndarray:
    """Radar-frame linear velocity at each frame by central differences (ends: one-sided)."""
    n = len(poses)
    v = np.zeros((n, 3))
    for k in range(n):
        a, b = max(k - 1, 0), min(k + 1, n - 1)
        v[k] = relative(poses[a], poses[b])[:3, 3] / (stamps[b] - stamps[a])
    return v


def lsq_velocity(pts: np.ndarray, keep_frac: float = 0.7, ridge: float = 1e-3) -> np.ndarray:
    """Classical baseline: least squares + one trimming pass (no learning)."""
    u, d = pts[:, :3], pts[:, 4]
    A = u.T @ u + ridge * np.eye(3)
    v = np.linalg.solve(A, u.T @ -d)
    res = np.abs(u @ v + d)
    m = res <= np.quantile(res, keep_frac)
    return np.linalg.solve(u[m].T @ u[m] + ridge * np.eye(3), u[m].T @ -d[m])


def weighted_solve(u, d, w, ridge: float = 1e-2):
    """Batched v = argmin sum w (u.v + d)^2 + ridge |v|^2. u (B,K,3), d (B,K), w (B,K)."""
    Wu = u * w[..., None]
    A = Wu.transpose(1, 2) @ u + ridge * torch.eye(3, device=u.device)
    b = (Wu * -d[..., None]).sum(1)
    return torch.linalg.solve(A, b)


class PointScorer(nn.Module):
    def __init__(self, cin: int, h: int = 64):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(cin, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        f = self.local(x)
        g = f.max(1, keepdim=True).values.expand_as(f)  # global context (PointNet)
        return torch.sigmoid(self.head(torch.cat([f, g], -1)).squeeze(-1))


class DopplerVelocityNet(nn.Module):
    def __init__(self, doppler_scale: float = 2.0, range_scale: float = 8.0):
        super().__init__()
        self.ds, self.rs = doppler_scale, range_scale
        self.score1 = PointScorer(6)
        self.score2 = PointScorer(7)

    def _feats(self, p):
        return torch.cat([p[..., :3], p[..., 3:4] / self.rs, p[..., 4:5] / self.ds, p[..., 5:6]], -1)

    def forward(self, p):
        u, d = p[..., :3], p[..., 4]
        x = self._feats(p)
        v1 = weighted_solve(u, d, self.score1(x))
        res = ((u * v1[:, None]).sum(-1) + d)[..., None] / self.ds
        w2 = self.score2(torch.cat([x, res], -1))
        return weighted_solve(u, d, w2), w2


def precompute(seq_dir: str | Path, grid: PolarGridCfg, k: int = 512) -> None:
    """Write points.npy (N, k, 6) and velocity.npy (N, 3) next to a processed sequence."""
    from .data.store import load_sequence
    s = load_sequence(seq_dir)
    pts = np.stack([extract_points(np.asarray(s.polar[i]), grid, k) for i in range(len(s))])
    np.save(Path(seq_dir) / "points.npy", pts)
    np.save(Path(seq_dir) / "velocity.npy", velocity_labels(s.poses, s.stamps).astype(np.float32))
