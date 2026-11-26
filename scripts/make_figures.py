"""Figures for the README: python scripts/make_figures.py --config configs/coloradar_4runs.yaml

Writes to docs/figures/:
  doppler_physics.png   measured vs physics-predicted Doppler per return, coloured by learned weight
  velocity.png          radar velocity over time on an unseen run: ground truth, least squares, learned
  learned_weights.gif   top view of each frame's returns, coloured by learned weight
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

from radar_odom.config import load_config
from radar_odom.doppler import DopplerVelocityNet, lsq_velocity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--doppler", default="runs/doppler_net")
    ap.add_argument("--seq", default="12_21_2020_ec_hallways_run4")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg.data.root) / a.seq
    pts = np.load(root / "points.npy")
    vel = np.load(root / "velocity.npy")
    t = np.load(root / "stamps.npy"); t = t - t[0]
    ck = torch.load(Path(a.doppler) / "best.pt", map_location="cpu")
    net = DopplerVelocityNet(ck["doppler_scale"]).eval()
    net.load_state_dict(ck["model"])
    with torch.no_grad():
        v_net, w = [x.numpy() for x in net(torch.from_numpy(pts))]
    v_lsq = np.stack([lsq_velocity(p) for p in pts])

    # 1. physics: measured Doppler vs -u.v_gt, all returns of 60 frames, coloured by learned weight
    sel = np.arange(60, 120)
    u, d = pts[sel, :, :3], pts[sel, :, 4]
    pred = -(u * vel[sel][:, None]).sum(-1)
    ww = w[sel]
    o = np.argsort(ww.ravel())
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    sc = ax.scatter(pred.ravel()[o], d.ravel()[o], c=ww.ravel()[o], s=4, cmap="viridis", vmin=0, vmax=1)
    lim = 1.6
    dmin = float(pts[..., 4].min())  # most negative Doppler the radar reports in this run
    ax.axhspan(-lim, dmin, color="0.88", zorder=0)
    ax.text(-0.1, dmin - 0.12, f"never measured (below {dmin:.2f} m/s)", fontsize=8, color="0.3")
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="static world: d = −u·v")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Doppler predicted from ground-truth velocity [m/s]")
    ax.set_ylabel("measured Doppler [m/s]")
    ax.set_title("Measured Doppler is quantized (0.165 m/s steps) and cut off\n"
                 "near −1.15 m/s, while the rig moves at ~1.25 m/s", fontsize=10)
    ax.text(-1.55, -1.52, f"{a.seq} (unseen)\n60 frames × 512 returns", fontsize=7, va="bottom")
    plt.colorbar(sc, ax=ax, label="learned trust weight")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "doppler_physics.png", dpi=130); plt.close(fig)

    # 2. velocity over time
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    for i, (axx, name) in enumerate(zip(axs, ["forward (x)", "sideways (y)"])):
        c_l = np.corrcoef(v_lsq[:, i], vel[:, i])[0, 1]
        c_n = np.corrcoef(v_net[:, i], vel[:, i])[0, 1]
        axx.plot(t, vel[:, i], "k", lw=2, label="ground truth")
        axx.plot(t, v_lsq[:, i], "C2", lw=1, alpha=0.8, label=f"Doppler least squares (r={c_l:.2f})")
        axx.plot(t, v_net[:, i], "C3", lw=1.2, label=f"learned Doppler (r={c_n:.2f})")
        axx.set_ylabel(f"{name} [m/s]"); axx.legend(loc="upper right", fontsize=8); axx.grid(alpha=0.3)
    axs[1].set_xlabel("time [s]")
    axs[0].set_title(f"Radar-frame velocity from a single frame, {a.seq} (never seen in training)")
    fig.tight_layout(); fig.savefig(out / "velocity.png", dpi=130); plt.close(fig)

    # 3. GIF: top view of the returns, coloured by learned weight
    frames = np.arange(40, 160)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    xy = lambda k: pts[k, :, :3] * pts[k, :, 3:4]
    s0 = ax.scatter([], [], c=[], s=[], cmap="viridis", vmin=0, vmax=1)
    ax.set_xlim(-7.6, 7.6); ax.set_ylim(0, 7.6); ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlabel("y [m] (left)"); ax.set_ylabel("x [m] (forward)")
    plt.colorbar(s0, ax=ax, label="learned trust weight", shrink=0.8)
    ax.text(7.4, 7.2, "black: true velocity   red: learned", fontsize=8, va="top")
    arrow = [None]
    title = ax.set_title("", fontsize=10)

    def draw(k):
        p = xy(k)
        s0.set_offsets(np.c_[p[:, 1], p[:, 0]])
        s0.set_array(w[k]); s0.set_sizes(6 + 30 * pts[k, :, 5])
        if arrow[0] is not None:
            arrow[0].remove()
        g, e = vel[k], v_net[k]
        arrow[0] = ax.quiver([0, 0], [0.2, 0.2], [g[1], e[1]], [g[0], e[0]], color=["k", "C3"],
                             angles="xy", scale_units="xy", scale=0.25, width=0.012)
        title.set_text(f"t = {t[k]:5.1f} s   speed: truth {np.linalg.norm(g[:2]):.2f}, "
                       f"learned {np.linalg.norm(e[:2]):.2f} m/s")
        return s0,

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=200)
    anim.save(out / "learned_weights.gif", writer=animation.PillowWriter(fps=5), dpi=80)
    plt.close(fig)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
