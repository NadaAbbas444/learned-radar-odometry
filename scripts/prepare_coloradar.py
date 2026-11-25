"""Convert ColoRadar sequences (plain-file layout) to the processed format.

python scripts/prepare_coloradar.py --raw /path/to/coloradar --calib /path/to/coloradar/calib \
    --out data/coloradar hallways_run0 hallways_run1 ...

Run scripts/check_coloradar.py on the first output before converting everything.
"""
import argparse
from pathlib import Path

from radar_odom.data.coloradar import load_sequence_raw
from radar_odom.data.store import save_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", default="EARC", help="on-disk axis order of a heatmap .bin")
    ap.add_argument("sequences", nargs="+")
    a = ap.parse_args()
    for name in a.sequences:
        polar, stamps, poses, grid = load_sequence_raw(Path(a.raw) / name, a.calib, layout=a.layout)
        save_sequence(Path(a.out) / name, polar, stamps, poses,
                      {"source": "coloradar", "layout": a.layout, "polar_grid": grid.__dict__})
        print(f"{name}: {len(stamps)} frames, grid {grid.shape}, range bin {grid.range_bin_width} m", flush=True)


if __name__ == "__main__":
    main()
