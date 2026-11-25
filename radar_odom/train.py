"""Train: python -m radar_odom.train --config configs/synthetic.yaml"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .config import load_config, to_dict
from .data.dataset import PairDataset, label_stats, load_split
from .model import RadarOdomNet


def weighted_mse(pred, target, rot_weight: float):
    w = torch.tensor([1, 1, 1, rot_weight, rot_weight, rot_weight], device=pred.device, dtype=pred.dtype)
    return (((pred - target) ** 2) * w).mean()


def run_epoch(model, loader, device, rot_weight, opt=None, scaler=None, amp=False):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                pred = model(x)
            loss = weighted_mse(pred.float(), y, rot_weight)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            total += loss.item() * len(x)  # counted once per batch
            n += len(x)
    return total / max(n, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=5")
    args = ap.parse_args(argv)
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)
    cfg = load_config(args.config, overrides)
    tc, dc = cfg.train, cfg.data

    torch.manual_seed(tc.seed); np.random.seed(tc.seed); random.seed(tc.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(tc.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))

    train_seqs, val_seqs = load_split(dc.root, dc.train), load_split(dc.root, dc.val)
    overlap = set(dc.train) & (set(dc.val) | set(dc.test))
    if overlap:
        raise ValueError(f"sequences in more than one split: {overlap}")
    mean, std = label_stats(train_seqs, dc.strides, dc.reverse_aug)
    kw = dict(label_mean=mean, label_std=std, use_doppler=dc.use_doppler, doppler_scale=dc.doppler_scale)
    ds_tr = PairDataset(train_seqs, dc.strides, reverse_aug=dc.reverse_aug, **kw)
    ds_va = PairDataset(val_seqs, dc.strides, **kw)
    dl_tr = DataLoader(ds_tr, tc.batch_size, shuffle=True, num_workers=tc.num_workers, pin_memory=True,
                       drop_last=True, persistent_workers=tc.num_workers > 0)
    dl_va = DataLoader(ds_va, tc.batch_size, num_workers=tc.num_workers, pin_memory=True,
                       persistent_workers=tc.num_workers > 0)

    in_ch = 2 * (2 if dc.use_doppler else 1)
    model = RadarOdomNet(cfg.polar, cfg.cartesian, in_ch, base=tc.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    amp = tc.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)

    # Exact val loss of the constant predictor "always output the training mean" (0 in
    # normalised units). A model is only learning something if it gets below this.
    yv = (ds_va.labels - mean) / std
    wv = np.array([1, 1, 1, tc.rot_weight, tc.rot_weight, tc.rot_weight])
    const_val = float((yv**2 * wv).mean())
    print(f"device={device} train_pairs={len(ds_tr)} val_pairs={len(ds_va)} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  "
          f"constant-predictor val loss={const_val:.4f}", flush=True)
    history, best, bad = [], float("inf"), 0
    for ep in range(1, tc.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, dl_tr, device, tc.rot_weight, opt, scaler, amp)
        va = run_epoch(model, dl_va, device, tc.rot_weight, amp=amp)
        sched.step(va)
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va, "lr": lr, "sec": time.time() - t0,
                        "const_val_loss": const_val})
        print(f"epoch {ep:3d}  train {tr:.4f}  val {va:.4f}  (constant {const_val:.4f})  lr {lr:.1e}  "
              f"{time.time()-t0:.0f}s", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if va < best:
            best, bad = va, 0
            torch.save({"model": model.state_dict(), "mean": mean, "std": std, "config": to_dict(cfg),
                        "epoch": ep, "val_loss": va}, out / "best.pt")
        else:
            bad += 1
            if bad >= tc.patience:
                print(f"early stop: no val improvement for {tc.patience} epochs", flush=True)
                break
    print(f"best val {best:.4f}; checkpoint {out/'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
