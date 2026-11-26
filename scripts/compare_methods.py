"""Compare CNN vs Doppler-based methods on the test runs.

python scripts/compare_methods.py --config configs/coloradar_4runs.yaml \
    --cnn runs/coloradar_4runs_dopmask --doppler runs/doppler_net --out runs/comparison

Per consecutive frame pair (k, k+1) on each test run:
  translation  t_k,k+1 from each method; Doppler methods use dt * (v_k + v_k+1) / 2
  rotation     only the CNN predicts it (Doppler carries none, see radar_odom/doppler.py)
Trajectories "with GT rotation" combine each method's translation with ground-truth
rotation, isolating translation quality; the CNN is also shown with its own rotation.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from radar_odom.config import Config, _build, load_config
from radar_odom.data.dataset import PairDataset, load_split, make_pairs, pair_labels
from radar_odom.doppler import DopplerVelocityNet, lsq_velocity
from radar_odom.evaluate import predict
from radar_odom.geometry import compose_trajectory, from_twist, rotation_angle, umeyama_se3
from radar_odom.model import RadarOdomNet


def ate(P, G):
    A = umeyama_se3(P, G)
    return float(np.sqrt((((P @ A[:3, :3].T + A[:3, 3]) - G) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cnn", required=True)
    ap.add_argument("--doppler", required=True)
    ap.add_argument("--out", default="runs/comparison")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(Path(a.cnn) / "best.pt", map_location="cpu", weights_only=False)
    ccfg: Config = _build(Config, ck["config"])
    cnn = RadarOdomNet(ccfg.polar, ccfg.cartesian, 4, base=ccfg.train.base_channels).to(dev)
    cnn.load_state_dict(ck["model"])
    dk = torch.load(Path(a.doppler) / "best.pt", map_location="cpu")
    dnet = DopplerVelocityNet(dk["doppler_scale"]).to(dev).eval()
    dnet.load_state_dict(dk["model"])

    train = load_split(cfg.data.root, cfg.data.train)
    const = pair_labels(train, make_pairs(train, [1])).mean(0)

    report, rows = {}, []
    names = cfg.data.test
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 6.5), squeeze=False)
    for ax, seq in zip(axes[0], load_split(cfg.data.root, names)):
        ds = PairDataset([seq], [1], ck["mean"], ck["std"], ccfg.data.use_doppler, ccfg.data.doppler_scale)
        gt = ds.labels
        T_gt = from_twist(gt)
        n = len(gt)
        cnn_tw = predict(cnn, ds, dev)
        pts = np.load(Path(cfg.data.root) / seq.name / "points.npy")
        dt = np.diff(seq.stamps)[:, None]
        v_lsq = np.stack([lsq_velocity(p) for p in pts])
        with torch.no_grad():
            v_net = np.concatenate([dnet(torch.from_numpy(pts[i:i + 256]).to(dev))[0].cpu().numpy()
                                    for i in range(0, len(pts), 256)])
        trans = {
            "zero": np.zeros((n, 3)),
            "constant": np.repeat(const[None, :3], n, 0),
            "CNN (stacked pair 3D CNN)": cnn_tw[:, :3],
            "Doppler least squares (no learning)": dt * (v_lsq[:-1] + v_lsq[1:]) / 2,
            "learned Doppler (proposed)": dt * (v_net[:-1] + v_net[1:]) / 2,
        }
        G = compose_trajectory(T_gt)[:, :3, 3]
        res = {}
        ax.plot(G[:, 0], G[:, 1], "k-", lw=2.5, label="ground truth")
        styles = {"zero": None, "constant": "C1--", "CNN (stacked pair 3D CNN)": "C0-",
                  "Doppler least squares (no learning)": "C2-", "learned Doppler (proposed)": "C3-"}
        for name, t in trans.items():
            err = np.linalg.norm(t - gt[:, :3], axis=1)
            T = T_gt.copy(); T[:, :3, 3] = t
            P = compose_trajectory(T)[:, :3, 3]
            length = np.linalg.norm(np.diff(G, axis=0), axis=1).sum()
            res[name] = {"trans_err_mean_m": float(err.mean()), "trans_err_median_m": float(np.median(err)),
                         "per_axis_mae_m": np.abs(t - gt[:, :3]).mean(0).round(4).tolist(),
                         "ate_gt_rot_m": ate(P, G),
                         "ate_gt_rot_pct_of_path": float(100 * ate(P, G) / length)}
            if styles[name]:
                ax.plot(P[:, 0], P[:, 1], styles[name], lw=1.4, label=f"{name}: ATE {res[name]['ate_gt_rot_m']:.2f} m")
        # rotation: only the CNN predicts it
        Rerr = lambda R: np.rad2deg(rotation_angle(np.swapaxes(R, 1, 2) @ T_gt[:, :3, :3])).mean()
        res["rotation_err_deg"] = {"CNN": float(Rerr(from_twist(cnn_tw)[:, :3, :3])),
                                   "zero": float(Rerr(np.repeat(np.eye(3)[None], n, 0))),
                                   "constant": float(Rerr(from_twist(np.repeat(const[None], n, 0))[:, :3, :3]))}
        Pc = compose_trajectory(from_twist(cnn_tw))[:, :3, 3]
        res["CNN own rotation"] = {"ate_m": ate(Pc, G)}
        ax.plot(Pc[:, 0], Pc[:, 1], "C0:", lw=1.2, label=f"CNN with its own rotation: ATE {ate(Pc, G):.2f} m")
        ax.set_title(f"{seq.name}  ({length:.0f} m, {n} steps)\ntranslation from each method + ground-truth rotation")
        ax.axis("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(fontsize=8, loc="best")
        report[seq.name] = res
        for name in trans:
            rows.append((seq.name, name, res[name]))
    fig.tight_layout(); fig.savefig(out / "trajectories.png", dpi=120)

    # bar chart of per-step translation error
    methods = list(trans)
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(names)
    for i, s in enumerate(names):
        ax.bar(np.arange(len(methods)) + i * w, [report[s][m]["trans_err_mean_m"] * 100 for m in methods], w,
               label=s)
    ax.set_xticks(np.arange(len(methods)) + w * (len(names) - 1) / 2)
    ax.set_xticklabels([m.split(" (")[0] for m in methods], rotation=15)
    ax.set_ylabel("mean translation error per step [cm]"); ax.legend(); ax.set_title("Unseen test runs")
    fig.tight_layout(); fig.savefig(out / "step_error.png", dpi=120)

    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    md = ["| test run | method | step trans err mean [cm] | median [cm] | ATE w/ GT rot [m] | ATE [% path] |",
          "|---|---|---|---|---|---|"]
    for s, m, r in rows:
        md.append(f"| {s} | {m} | {100*r['trans_err_mean_m']:.1f} | {100*r['trans_err_median_m']:.1f} | "
                  f"{r['ate_gt_rot_m']:.2f} | {r['ate_gt_rot_pct_of_path']:.1f} |")
    md += ["", "| test run | rotation err per step [deg]: CNN | zero | constant | CNN own-rotation ATE [m] |",
           "|---|---|---|---|---|"]
    for s in names:
        r = report[s]["rotation_err_deg"]
        md.append(f"| {s} | {r['CNN']:.2f} | {r['zero']:.2f} | {r['constant']:.2f} | "
                  f"{report[s]['CNN own rotation']['ate_m']:.2f} |")
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
