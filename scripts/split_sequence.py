"""Cut one processed sequence into consecutive, non-overlapping chunks (stand-alone sequences).

python scripts/split_sequence.py data/coloradar/12_21_2020_ec_hallways_run0 --fractions 0.7 0.15 0.15 --gap 5

Only for a first look when a single run is available: chunks of the same run share the
environment (and possibly revisited places), so results are optimistic compared with
held-out runs.
"""
import argparse
from pathlib import Path

import numpy as np

from radar_odom.data.store import load_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.7, 0.15, 0.15])
    ap.add_argument("--gap", type=int, default=5, help="frames dropped between chunks")
    ap.add_argument("--names", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    src = Path(a.seq)
    s = load_sequence(src)
    n = len(s)
    edges = np.r_[0, np.round(np.cumsum(a.fractions) / sum(a.fractions) * n).astype(int)]
    for name, lo, hi in zip(a.names, edges[:-1], edges[1:]):
        lo2, hi2 = lo + (a.gap if lo > 0 else 0), hi
        out = src.parent / f"{src.name}_{name}"
        out.mkdir(exist_ok=True)
        # polar is already stored as log1p intensity: copy the slice as is
        np.save(out / "polar.npy", np.asarray(s.polar[lo2:hi2]))
        np.save(out / "stamps.npy", s.stamps[lo2:hi2])
        np.save(out / "poses.npy", s.poses[lo2:hi2])
        (out / "meta.json").write_text(f'{{"source": "{src.name}", "frames": [{lo2}, {hi2}], "intensity": "log1p"}}')
        print(f"{out.name}: frames {lo2}-{hi2} ({hi2 - lo2})")


if __name__ == "__main__":
    main()
