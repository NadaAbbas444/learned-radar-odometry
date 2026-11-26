"""Visual check of one processed sequence: polar slice, Cartesian top view, trajectory.

python scripts/check_coloradar.py --config configs/coloradar.yaml --seq hallways_run0 --frame 100

If the heatmap layout or bin angles are wrong, walls in the top view look curved or
smeared, and the scene will not line up with the direction of travel.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from radar_odom.config import load_config
from radar_odom.data.store import load_sequence
from radar_odom.encoding import encode_numpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="check.png")
    a = ap.parse_args()
    cfg = load_config(a.config)
    s = load_sequence(Path(cfg.data.root) / a.seq)
    P = np.asarray(s.polar[a.frame], dtype=np.float32)
    C = encode_numpy(np.maximum(P, 0), cfg.polar, cfg.cartesian)  # intensity stored as log1p
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].imshow(P[0].max(0), aspect="auto", origin="lower")
    ax[0].set_title("polar: max over elevation"); ax[0].set_xlabel("range bin"); ax[0].set_ylabel("azimuth bin")
    c = cfg.cartesian
    ext = [-c.ny / 2 * c.voxel, c.ny / 2 * c.voxel, c.x_min, c.x_min + c.nx * c.voxel]
    ax[1].imshow(C[0].max(2), origin="lower", extent=ext, aspect="equal")
    ax[1].set_title("Cartesian top view (x up, y left)"); ax[1].set_xlabel("y [m]"); ax[1].set_ylabel("x [m]")
    ax[1].invert_xaxis()
    p = s.poses[:, :3, 3]
    ax[2].plot(p[:, 0], p[:, 1]); ax[2].plot(*p[a.frame, :2], "ro"); ax[2].axis("equal")
    ax[2].set_title(f"ground truth, {len(s)} frames")
    fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
