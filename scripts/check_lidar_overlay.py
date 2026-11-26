"""Overlay a lidar scan on the radar image to check radar geometry and calibration.

python scripts/check_lidar_overlay.py --raw data/coloradar_raw --seq 12_21_2020_ec_hallways_run0 \
    --lidar-index 600 --config configs/coloradar.yaml

Lidar points (x, y, z, intensity float32) are moved into the radar frame with the calib
transforms and compared with the radar slice at sensor height. The score is the mean
radar value at lidar-occupied cells minus the mean elsewhere (in standard deviations),
for the image as encoded and for mirrored/rotated variants. The encoding is right when
the unmodified variant scores clearly highest.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from radar_odom.config import load_config
from radar_odom.data.coloradar import read_heatmap, read_heatmap_cfg, read_transform
from radar_odom.encoding import encode_numpy, voxel_centres
from radar_odom.geometry import inv


def score(radar2d, occ):
    z = (radar2d - radar2d.mean()) / (radar2d.std() + 1e-9)
    return float(z[occ].mean() - z[~occ].mean()) if occ.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--lidar-index", type=int, default=600)
    ap.add_argument("--config", required=True)
    ap.add_argument("--zband", type=float, default=0.4, help="half-height of the slice (m)")
    ap.add_argument("--out", default="logs/lidar_overlay.png")
    a = ap.parse_args()
    cfg = load_config(a.config)
    raw, seq = Path(a.raw), Path(a.raw) / a.seq
    calib = raw / "calib"
    grid = read_heatmap_cfg(calib / "cascade" / "heatmap_cfg.txt")
    cfg.polar = grid

    lid_t = np.loadtxt(seq / "lidar" / "timestamps.txt")[a.lidar_index]
    hm_t = np.loadtxt(seq / "cascade" / "heatmaps" / "timestamps.txt")
    k = int(np.argmin(abs(hm_t - lid_t)))
    print(f"lidar scan {a.lidar_index} at t={lid_t:.3f}; radar frame {k}, dt={hm_t[k]-lid_t:+.3f}s")

    pts = np.fromfile(seq / "lidar" / "pointclouds" / f"lidar_pointcloud_{a.lidar_index}.bin",
                      np.float32).reshape(-1, 4)[:, :3]
    pts = pts[np.linalg.norm(pts, axis=1) > 0.3]
    T_radar_lidar = inv(read_transform(calib / "transforms" / "base_to_cascade.txt")) @ \
        read_transform(calib / "transforms" / "base_to_lidar.txt")
    p = pts @ T_radar_lidar[:3, :3].T + T_radar_lidar[:3, 3]

    P = read_heatmap(seq / "cascade" / "heatmaps" / "data" / f"heatmap_{k}.bin", grid)
    P[0] = np.log1p(np.maximum(P[0], 0))
    C = encode_numpy(P[:1], grid, cfg.cartesian)[0]  # (nx, ny, nz)
    zc = voxel_centres(cfg.cartesian)[0, 0, :, 2]
    sl = C[:, :, np.abs(zc) < a.zband].mean(2)
    # remove the range-dependent background (rings) so structure stands out
    xy = voxel_centres(cfg.cartesian)[:, :, 0, :2]
    rr = np.linalg.norm(xy, axis=-1)
    bins = np.digitize(rr, np.arange(0, 12, 0.12))
    bg = np.zeros_like(sl)
    for b in np.unique(bins):
        m = (bins == b) & (sl > 0)
        if m.any():
            bg[bins == b] = np.median(sl[m])
    sl = np.where(sl > 0, sl - bg, 0)

    c = cfg.cartesian
    occ = np.zeros(sl.shape, bool)
    band = p[np.abs(p[:, 2]) < a.zband]
    ix = np.floor((band[:, 0] - c.x_min) / c.voxel).astype(int)
    iy = np.floor(band[:, 1] / c.voxel + c.ny / 2).astype(int)
    ok = (ix >= 0) & (ix < c.nx) & (iy >= 0) & (iy < c.ny)
    occ[ix[ok], iy[ok]] = True
    fov = C[:, :, np.abs(zc) < a.zband].mean(2) > 0
    occ &= fov
    variants = {"as encoded": sl, "mirror y (azimuth sign flipped)": sl[:, ::-1]}
    for name, img in variants.items():
        print(f"score {name:32s} {score(img[fov], occ[fov]):+.3f}")

    ext = [-c.ny / 2 * c.voxel, c.ny / 2 * c.voxel, c.x_min, c.x_min + c.nx * c.voxel]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for axx, (name, img) in zip(ax, variants.items()):
        axx.imshow(np.ma.masked_where(~fov, img), origin="lower", extent=ext, cmap="viridis")
        axx.plot(band[ok][:, 1], band[ok][:, 0], ".", color="red", ms=1.0, alpha=0.6)
        axx.set_xlim(ext[1], ext[0]); axx.set_ylim(ext[2], ext[3])
        axx.set_title(f"{name}: score {score(img[fov], occ[fov]):+.2f}")
        axx.set_xlabel("y [m] (left)"); axx.set_ylabel("x [m] (forward)")
    fig.suptitle(f"radar slice |z|<{a.zband} m (range background removed) + lidar (red)")
    fig.tight_layout(); Path(a.out).parent.mkdir(exist_ok=True, parents=True); fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
