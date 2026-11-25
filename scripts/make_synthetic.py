"""Generate synthetic sequences: python scripts/make_synthetic.py --config configs/synthetic.yaml"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from radar_odom.config import load_config
from radar_odom.data.store import save_sequence
from radar_odom.data.synthetic import make_sequence


def one(args):
    name, n, grid, seed, root = args
    polar, stamps, poses = make_sequence(n, grid, seed)
    save_sequence(Path(root) / name, polar, stamps, poses, {"source": "synthetic", "seed": seed})
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(a.config)
    names = cfg.data.train + cfg.data.val + cfg.data.test
    jobs = [(n, a.frames, cfg.polar, 1000 + i, cfg.data.root) for i, n in enumerate(names)]
    with ProcessPoolExecutor(a.workers) as ex:
        for name in ex.map(one, jobs):
            print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
