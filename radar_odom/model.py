"""Stacked-pair 3D CNN (after Kramer's radar scan alignment network).

Input: two polar frames stacked on channels, (B, 2*C, El, Az, R). They are encoded to
Cartesian voxels on the GPU, passed through 3D conv blocks and regressed to a
normalised 6-vector [t, rotvec] of T_1_2.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import CartesianGridCfg, PolarGridCfg
from .encoding import PolarToCartesian


def conv_block(cin: int, cout: int, n: int = 2) -> nn.Sequential:
    layers = []
    for _ in range(n):
        layers += [nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True)]
        cin = cout
    return nn.Sequential(*layers)


class RadarOdomNet(nn.Module):
    def __init__(self, polar: PolarGridCfg, cart: CartesianGridCfg, in_channels: int, base: int = 16,
                 n_blocks: int = 4):
        super().__init__()
        self.encode = PolarToCartesian(polar, cart)
        blocks, size, c = [], [cart.nx, cart.ny, cart.nz], in_channels
        for i in range(n_blocks):
            cout = base * min(2**i, 8)
            k = tuple(2 if s >= 2 else 1 for s in size)
            blocks += [conv_block(c, cout), nn.MaxPool3d(k, k)]
            size = [s // kk for s, kk in zip(size, k)]
            c = cout
        self.features = nn.Sequential(*blocks)
        flat = c * size[0] * size[1] * size[2]
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 6))

    def forward(self, polar_pair: torch.Tensor) -> torch.Tensor:
        x = self.encode(polar_pair)
        return self.head(self.features(x))
