"""Typed configuration loaded from YAML. No paths are hard-coded anywhere else."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import yaml


@dataclass
class PolarGridCfg:
    """Polar heatmap geometry. Stored arrays are always (C, elevation, azimuth, range).

    If `heatmap_cfg` is set (path to ColoRadar calib/cascade/heatmap_cfg.txt), the bin
    counts, range bin width and the real (non-uniform) bin angles are read from it.
    """
    heatmap_cfg: Optional[str] = None
    num_range_bins: int = 128
    range_bin_width: float = 0.125  # metres per range bin
    range_offset: float = 0.0  # range of bin 0 (metres)
    # Either explicit bin centre angles (radians, increasing) or a uniform FOV.
    azimuth_bins: Optional[List[float]] = None
    elevation_bins: Optional[List[float]] = None
    num_azimuth_bins: int = 128
    num_elevation_bins: int = 32
    azimuth_fov_deg: float = 140.0
    elevation_fov_deg: float = 40.0

    def azimuths(self) -> np.ndarray:
        if self.azimuth_bins is not None:
            return np.asarray(self.azimuth_bins, dtype=np.float64)
        h = np.deg2rad(self.azimuth_fov_deg) / 2
        return np.linspace(-h, h, self.num_azimuth_bins)

    def elevations(self) -> np.ndarray:
        if self.elevation_bins is not None:
            return np.asarray(self.elevation_bins, dtype=np.float64)
        h = np.deg2rad(self.elevation_fov_deg) / 2
        return np.linspace(-h, h, self.num_elevation_bins)

    def ranges(self) -> np.ndarray:
        return self.range_offset + self.range_bin_width * np.arange(self.num_range_bins)

    @property
    def shape(self):
        return (len(self.elevations()), len(self.azimuths()), self.num_range_bins)


@dataclass
class CartesianGridCfg:
    """Output voxel grid in the radar frame (x forward, y left, z up).

    x spans [x_min, x_min + nx*voxel); y and z are centred on the sensor.
    """
    nx: int = 64
    ny: int = 128
    nz: int = 64
    voxel: float = 0.12
    x_min: float = 0.0


@dataclass
class DataCfg:
    root: str = "data/processed"
    train: List[str] = field(default_factory=list)
    val: List[str] = field(default_factory=list)
    test: List[str] = field(default_factory=list)
    strides: List[int] = field(default_factory=lambda: [1])  # frame gaps used to build pairs
    eval_stride: int = 1
    use_doppler: bool = True
    doppler_scale: float = 2.0  # m/s mapped to 1.0
    reverse_aug: bool = True  # also train on (k+n, k) with the inverted label


@dataclass
class TrainCfg:
    out_dir: str = "runs/default"
    epochs: int = 40
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 8  # early stopping on validation loss
    rot_weight: float = 1.0  # loss weight on (normalised) rotation terms
    num_workers: int = 4
    amp: bool = True
    seed: int = 0
    base_channels: int = 16


@dataclass
class Config:
    polar: PolarGridCfg = field(default_factory=PolarGridCfg)
    cartesian: CartesianGridCfg = field(default_factory=CartesianGridCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


def _build(cls, d: dict | None):
    d = d or {}
    kwargs: dict[str, Any] = {}
    names = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(names)
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    for k, v in d.items():
        ft = names[k].type
        sub = {"PolarGridCfg": PolarGridCfg, "CartesianGridCfg": CartesianGridCfg,
               "DataCfg": DataCfg, "TrainCfg": TrainCfg}.get(str(ft))
        kwargs[k] = _build(sub, v) if sub is not None else v
    return cls(**kwargs)


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    for k, v in (overrides or {}).items():
        sec, key = k.split(".", 1)
        d.setdefault(sec, {})[key] = v
    cfg = _build(Config, d)
    if cfg.polar.heatmap_cfg:
        from .data.coloradar import read_heatmap_cfg  # local import: avoid cycle
        path = cfg.polar.heatmap_cfg
        cfg.polar = read_heatmap_cfg(Path(path))
        cfg.polar.heatmap_cfg = path
    return cfg


def to_dict(obj) -> Any:
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj
