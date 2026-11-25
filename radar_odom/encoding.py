"""Polar -> Cartesian radar image encoding (trilinear), vectorised on the GPU.

The sampling grid is computed once from the sensor geometry: for each Cartesian voxel
centre (x, y, z) in the radar frame we get

    r = |(x, y, z)|,  azimuth = atan2(y, x),  elevation = atan2(z, hypot(x, y))

and convert each to a *fractional bin index* using the real bin centres (which may be
non-uniform, as in ColoRadar). Voxels outside the field of view or range sample zero.
`torch.nn.functional.grid_sample` then does the trilinear interpolation for a whole
batch in one call.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CartesianGridCfg, PolarGridCfg


def voxel_centres(cart: CartesianGridCfg) -> np.ndarray:
    """(nx, ny, nz, 3) voxel centre coordinates in metres (radar frame)."""
    s = cart.voxel
    x = cart.x_min + (np.arange(cart.nx) + 0.5) * s
    y = (np.arange(cart.ny) - cart.ny / 2 + 0.5) * s
    z = (np.arange(cart.nz) - cart.nz / 2 + 0.5) * s
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)


def _frac_index(values: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Fractional index of `values` along monotonically increasing bin centres; NaN outside."""
    idx = np.interp(values, centres, np.arange(len(centres)), left=np.nan, right=np.nan)
    return idx


def sampling_grid(polar: PolarGridCfg, cart: CartesianGridCfg) -> np.ndarray:
    """Normalised grid for grid_sample, shape (nz, ny, nx, 3) -> output volume (z, y, x).

    The polar input to grid_sample is laid out (C, El, Az, R) = (C, D, H, W), so the grid's
    last dim is (range, azimuth, elevation) in [-1, 1] (align_corners=True).
    """
    p = voxel_centres(cart)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.hypot(x, y))

    ri = _frac_index(r, polar.ranges())
    ai = _frac_index(az, polar.azimuths())
    ei = _frac_index(el, polar.elevations())
    n_e, n_a, n_r = polar.shape

    def norm(i, n):
        return 2.0 * i / (n - 1) - 1.0

    g = np.stack([norm(ri, n_r), norm(ai, n_a), norm(ei, n_e)], axis=-1)
    g[~np.isfinite(g).all(-1)] = 2.0  # outside -> zero padding
    # (nx, ny, nz, 3) -> (nz, ny, nx, 3) so output is (C, z, y, x)
    return np.ascontiguousarray(g.transpose(2, 1, 0, 3)).astype(np.float32)


class PolarToCartesian(nn.Module):
    """(B, C, El, Az, R) polar images -> (B, C, nx, ny, nz) Cartesian images."""

    def __init__(self, polar: PolarGridCfg, cart: CartesianGridCfg):
        super().__init__()
        self.register_buffer("grid", torch.from_numpy(sampling_grid(polar, cart))[None], persistent=False)

    def forward(self, polar_img: torch.Tensor) -> torch.Tensor:
        B = polar_img.shape[0]
        out = F.grid_sample(polar_img.float(), self.grid.expand(B, -1, -1, -1, -1),
                            mode="bilinear", padding_mode="zeros", align_corners=True)
        return out.permute(0, 1, 4, 3, 2)  # (B, C, z, y, x) -> (B, C, x, y, z)


def encode_numpy(polar_img: np.ndarray, polar: PolarGridCfg, cart: CartesianGridCfg) -> np.ndarray:
    """Convenience CPU version for plotting/tests. polar_img: (C, El, Az, R)."""
    enc = PolarToCartesian(polar, cart)
    with torch.no_grad():
        return enc(torch.from_numpy(np.ascontiguousarray(polar_img))[None].float())[0].numpy()
