"""Train DopplerVelocityNet: python -m radar_odom.train_doppler --config configs/coloradar_4runs.yaml

Needs points.npy / velocity.npy per sequence (radar_odom.doppler.precompute).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .doppler import DopplerVelocityNet


def load(root, names):
    P = np.concatenate([np.load(Path(root) / n / "points.npy") for n in names])
    V = np.concatenate([np.load(Path(root) / n / "velocity.npy") for n in names])
    return torch.from_numpy(P), torch.from_numpy(V)


def yaw_aug(p, v):
    """Rotate directions and velocity together about z: Doppler (-u.v) is unchanged."""
    a = torch.rand(len(p), device=p.device) * 2 * np.pi
    c, s = torch.cos(a), torch.sin(a)
    R = torch.zeros(len(p), 3, 3, device=p.device)
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1
    p = p.clone()
    p[..., :3] = p[..., :3] @ R.transpose(1, 2)
    return p, (R @ v[..., None])[..., 0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/doppler_net")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--no-aug", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Ptr, Vtr = [t.to(dev) for t in load(cfg.data.root, cfg.data.train)]
    Pva, Vva = [t.to(dev) for t in load(cfg.data.root, cfg.data.val)]
    net = DopplerVelocityNet(cfg.data.doppler_scale).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    zero_val = float((Vva ** 2).sum(1).mean())
    print(f"train frames {len(Ptr)}  val frames {len(Pva)}  params {sum(p.numel() for p in net.parameters())}  "
          f"zero-velocity val MSE {zero_val:.4f}", flush=True)
    best, bad, hist = float("inf"), 0, []
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(len(Ptr), device=dev)
        tl = 0.0
        for i in range(0, len(perm), 32):
            j = perm[i:i + 32]
            p, v = (Ptr[j], Vtr[j]) if a.no_aug else yaw_aug(Ptr[j], Vtr[j])
            loss = ((net(p)[0] - v) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(j)
        net.eval()
        with torch.no_grad():
            vl = float(((net(Pva)[0] - Vva) ** 2).sum(1).mean())
        hist.append({"epoch": ep, "train": tl / len(Ptr), "val": vl, "zero_val": zero_val})
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep:3d} train {tl/len(Ptr):.4f} val {vl:.4f} (zero {zero_val:.4f})", flush=True)
        if vl < best:
            best, bad = vl, 0
            torch.save({"model": net.state_dict(), "doppler_scale": cfg.data.doppler_scale}, out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop at epoch {ep}", flush=True)
                break
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    print(f"best val {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
