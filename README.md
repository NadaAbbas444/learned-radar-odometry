# radar-odom: learned alignment of dense 3D radar images

Estimate the 6-DoF motion between two radar scans with a 3D CNN, on the
[ColoRadar](https://arpg.colorado.edu/coloradar/) cascade-radar heatmaps. Two polar
heatmaps are stacked, resampled to a Cartesian voxel grid on the GPU, and regressed to
the relative pose `T_1_2` (frame 2 in frame 1), following Kramer's scan-alignment network.

This is a rewrite of `asldoc-2022-RobotX-Nada-Learning-Radar-Odometry`. The idea is kept;
the learning problem and the evaluation are rebuilt. See [What changed and why](#what-changed-and-why).

## Install

```bash
pip install -e ".[test]"
pytest
```

Python ≥ 3.9, PyTorch, NumPy, SciPy. No ROS needed.

## Layout

```
radar_odom/
  geometry.py        SE(3): relative poses, twists, composition, SLERP, Umeyama
  encoding.py        polar -> Cartesian trilinear resampling (grid_sample, real bin angles)
  model.py           stacked-pair 3D CNN
  train.py           training (split by sequence, early stopping on validation)
  evaluate.py        chained trajectories, RPE / ATE / drift vs zero & constant baselines
  data/
    store.py         processed sequence format (polar.npy, stamps.npy, poses.npy)
    sync.py          radar <-> ground-truth association by timestamp (interpolated)
    dataset.py       (k, k+s) pairs within a sequence, body-frame labels
    coloradar.py     reader for ColoRadar plain files
    synthetic.py     synthetic scenes with exact motion (pipeline check)
configs/             synthetic.yaml, coloradar.yaml
scripts/             make_synthetic.py, prepare_coloradar.py, check_coloradar.py
tests/               geometry, encoding, sync, dataset, model
```

## Quick start (synthetic, ~1 GB, minutes on a laptop GPU)

```bash
python scripts/make_synthetic.py --config configs/synthetic.yaml
python -m radar_odom.train --config configs/synthetic.yaml
python -m radar_odom.evaluate --run runs/synthetic
```

Synthetic data checks the pipeline end to end with exact ground truth. Its numbers say
nothing about accuracy on real radar.

## ColoRadar

1. Download sequences (one zip each) from the dataset page:
   <https://arpg.colorado.edu/coloradar/>, which links to the file share. Also get the
   `calib` folder. Skip the `longboard` runs (much faster motion than the rest).
2. Convert:
   ```bash
   python scripts/prepare_coloradar.py --raw /path/to/coloradar --calib /path/to/coloradar/calib \
       --out data/coloradar 12_21_2020_ec_hallways_run0 ...
   ```
3. **Check one sequence before converting everything:**
   ```bash
   python scripts/check_coloradar.py --config configs/coloradar.yaml --seq 12_21_2020_ec_hallways_run0 --frame 100
   ```
   Walls must look straight in the Cartesian top view. The reader in
   `radar_odom/data/coloradar.py` follows the paper (heatmap = elevation × azimuth ×
   range × [intensity, Doppler], float32) but file names and calib formats were written
   without the data at hand. Fix them there if the check fails.
4. Put the sequence names in `configs/coloradar.yaml` (split by sequence, ideally hold out
   whole runs of each environment for test), then train and evaluate as above.

## Reading results

`evaluate` writes `results_test.md`, `metrics_test.json`, `trajectories_test.png` and
`loss.png` into the run folder. Every metric is shown next to:

* **zero**: predicts no motion;
* **constant**: predicts the average training motion for every pair.

Unless the model clearly beats **constant**, it has not learned anything from the radar.
`train` prints the exact constant-predictor validation loss at the start for the same reason.

## What changed and why

| Old pipeline | Problem | Now |
|---|---|---|
| label `p2 - p1` (world frame), `q1·q2⁻¹` | depends on the absolute heading, which a sensor-frame image pair cannot see, so it can't be learned | `T_1_2 = T_w1⁻¹ T_w2`; test checks invariance to the world frame |
| all pairs `i<j` within 0.8 m | includes revisits, not odometry; each frame in dozens of pairs | `(k, k+s)` within one sequence, `s` in `data.strides` |
| random split over pairs | the same frames end up in train and validation | split by whole sequence, checked at start |
| training reshape `(2,64,128,64)` of a `(64,128,64,2)` array | input channels scrambled | channels kept explicit, Doppler used |
| Cartesian grid from 0 on every axis; angles mapped over 360°/180° | only one octant; wrong geometry | centred grid, real bin angles, out-of-FOV = 0; tests place points left/right/below |
| polar reshape `(N_r, N_a, N_e)` | ColoRadar stores `(N_e, N_a, N_r, 2)` | layout explicit (`--layout`) |
| pose file named by `list_b[i]`, paired by sorted index | poses can be attached to the wrong images | one array per sequence, interpolated at each heatmap time |
| trajectory = sum of 6-vectors | not a trajectory, even for ground truth | SE(3) composition, RPE/ATE/drift |
| loss counted twice, LR schedule driven by an `epoch` argument, early stop on train loss | misleading curves | fixed |
| text files per frame, Python triple loop | slow, huge | float16 `.npy` memmaps, encoding on GPU |
