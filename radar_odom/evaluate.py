"""Evaluate on held-out sequences: python -m radar_odom.evaluate --run runs/synthetic

For each test sequence, consecutive frames (k, k+eval_stride) are predicted, chained with
proper SE(3) composition, and compared with ground truth. Every metric is reported next
to two trivial baselines so a model that only learned the average motion is obvious:

  zero      predicts no motion
  constant  predicts the mean training motion for every pair
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import _build, Config
from .data.dataset import PairDataset, load_split, make_pairs, pair_labels
from .geometry import compose_trajectory, from_twist, inv, rotation_angle, umeyama_se3
from .model import RadarOdomNet


def trajectory_metrics(T_rel_pred: np.ndarray, T_rel_gt: np.ndarray) -> dict:
    E = inv(T_rel_pred) @ T_rel_gt
    t_err = np.linalg.norm(E[:, :3, 3], axis=1)
    r_err = np.rad2deg(rotation_angle(E[:, :3, :3]))
    P = compose_trajectory(T_rel_pred)[:, :3, 3]
    G = compose_trajectory(T_rel_gt)[:, :3, 3]
    A = umeyama_se3(P, G)
    P_al = P @ A[:3, :3].T + A[:3, 3]
    length = np.linalg.norm(np.diff(G, axis=0), axis=1).sum()
    return {
        "rpe_trans_m": float(t_err.mean()),
        "rpe_rot_deg": float(r_err.mean()),
        "ate_rmse_m": float(np.sqrt(((P_al - G) ** 2).sum(1).mean())),
        "end_drift_pct": float(100 * np.linalg.norm(P[-1] - G[-1]) / max(length, 1e-9)),
        "path_length_m": float(length),
    }


@torch.no_grad()
def predict(model, ds, device, batch_size=16):
    model.eval()
    out = []
    for x, _ in DataLoader(ds, batch_size, num_workers=4):
        out.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(out) * ds.std + ds.mean


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args(argv)
    run = Path(args.run)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    cfg: Config = _build(Config, ck["config"])
    dc = cfg.data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarOdomNet(cfg.polar, cfg.cartesian, 2 * (2 if dc.use_doppler else 1),
                         base=cfg.train.base_channels).to(device)
    model.load_state_dict(ck["model"])

    train_seqs = load_split(dc.root, dc.train)
    const = pair_labels(train_seqs, make_pairs(train_seqs, [dc.eval_stride])).mean(0)
    names = dc.test if args.split == "test" else dc.val
    results, rows = {}, []
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False)
    for ax, seq in zip(axes[0], load_split(dc.root, names)):
        ds = PairDataset([seq], [dc.eval_stride], ck["mean"], ck["std"], dc.use_doppler, dc.doppler_scale)
        gt = ds.labels
        pred = predict(model, ds, device)
        T_gt = from_twist(gt)
        res = {
            "model": trajectory_metrics(from_twist(pred), T_gt),
            "zero": trajectory_metrics(np.repeat(np.eye(4)[None], len(gt), 0), T_gt),
            "constant": trajectory_metrics(from_twist(np.repeat(const[None], len(gt), 0)), T_gt),
            "per_axis_mae": {"model": np.abs(pred - gt).mean(0).round(5).tolist(),
                             "constant": np.abs(const - gt).mean(0).round(5).tolist(),
                             "axes": ["tx", "ty", "tz", "rx", "ry", "rz"]},
        }
        results[seq.name] = res
        for name in ("model", "constant", "zero"):
            rows.append((seq.name, name, res[name]))
        G = compose_trajectory(T_gt)[:, :3, 3]
        ax.plot(G[:, 0], G[:, 1], "k-", lw=2, label="ground truth")
        for lab, tw, st in (("model", pred, "C0-"), ("constant", np.repeat(const[None], len(gt), 0), "C1--")):
            P = compose_trajectory(from_twist(tw))[:, :3, 3]
            ax.plot(P[:, 0], P[:, 1], st, label=lab)
        ax.set_title(seq.name); ax.axis("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend()
    fig.suptitle("Trajectories from chained relative poses (start frame = origin)")
    fig.tight_layout(); fig.savefig(run / f"trajectories_{args.split}.png", dpi=120)

    (run / f"metrics_{args.split}.json").write_text(json.dumps(results, indent=2))
    hdr = "| sequence | method | RPE trans [m] | RPE rot [deg] | ATE RMSE [m] | end drift [%] |\n|---|---|---|---|---|---|\n"
    body = "".join(f"| {s} | {m} | {r['rpe_trans_m']:.4f} | {r['rpe_rot_deg']:.3f} | {r['ate_rmse_m']:.3f} | "
                   f"{r['end_drift_pct']:.1f} |\n" for s, m, r in rows)
    (run / f"results_{args.split}.md").write_text(hdr + body)
    print(hdr + body)

    hist = json.loads((run / "history.json").read_text())
    plt.figure(figsize=(6, 4))
    plt.plot([h["train_loss"] for h in hist], label="train")
    plt.plot([h["val_loss"] for h in hist], label="val")
    plt.axhline(hist[0].get("const_val_loss", 1.0), color="gray", ls=":", label="constant predictor (val)")
    plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("normalised MSE"); plt.legend(); plt.tight_layout()
    plt.savefig(run / "loss.png", dpi=120)


if __name__ == "__main__":
    main()
