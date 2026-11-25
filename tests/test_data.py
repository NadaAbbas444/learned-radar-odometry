import numpy as np
import torch

from radar_odom.config import CartesianGridCfg, PolarGridCfg
from radar_odom.data.dataset import PairDataset, label_stats, make_pairs
from radar_odom.data.store import load_sequence, save_sequence
from radar_odom.data.sync import synchronise
from radar_odom.data.synthetic import make_sequence
from radar_odom.geometry import from_twist, inv, relative
from radar_odom.model import RadarOdomNet

GRID = PolarGridCfg(num_range_bins=32, range_bin_width=0.4, range_offset=0.1, num_azimuth_bins=32,
                    num_elevation_bins=8)


def test_sync_matches_by_time_not_order():
    gt_t = np.array([0.0, 0.1, 0.2, 0.3, 1.0, 1.1])
    gt = np.repeat(np.eye(4)[None], 6, 0)
    gt[:, 0, 3] = gt_t * 10  # x = 10 * t
    radar_t = np.array([0.25, 0.05, 0.6, 2.0])  # unsorted on purpose; 0.6 is in a gap, 2.0 outside
    keep, P = synchronise(radar_t, gt_t[::-1].copy(), gt[::-1].copy(), max_gap=0.2)
    assert keep.tolist() == [True, True, False, False]
    np.testing.assert_allclose(P[:, 0, 3], [2.5, 0.5], atol=1e-9)


def _seq(tmp_path, name, n=12, seed=0):
    polar, stamps, poses = make_sequence(n, GRID, seed)
    save_sequence(tmp_path / name, polar, stamps, poses)
    return load_sequence(tmp_path / name)


def test_pairs_are_within_sequence_and_labels_in_body_frame(tmp_path):
    s = [_seq(tmp_path, "a"), _seq(tmp_path, "b", seed=1)]
    P = make_pairs(s, [1, 2])
    assert ((P[:, 2] - P[:, 1]) >= 1).all() and set(P[:, 0]) == {0, 1}
    mean, std = label_stats(s, [1, 2], reverse_aug=False)
    ds = PairDataset(s, [1, 2], mean, std)
    i = 5
    si, a, b = ds.pairs[i]
    T = from_twist(ds.labels[i])
    np.testing.assert_allclose(s[si].poses[a] @ T, s[si].poses[b], atol=1e-6)
    x, y = ds[i]
    assert x.shape == (4,) + GRID.shape and torch.isfinite(x).all()
    np.testing.assert_allclose(y.numpy() * std + mean, ds.labels[i], atol=1e-5)


def test_reverse_aug_inverts_label(tmp_path):
    s = [_seq(tmp_path, "a")]
    mean, std = np.zeros(6), np.ones(6)
    ds = PairDataset(s, [1], mean, std, reverse_aug=True)
    np.random.seed(0)
    seen = set()
    for _ in range(30):
        x, y = ds[0]
        T = from_twist(y.numpy().astype(np.float64))
        fwd = relative(s[0].poses[0], s[0].poses[1])
        if np.allclose(T, fwd, atol=1e-5):
            seen.add("fwd")
        elif np.allclose(T, inv(fwd), atol=1e-5):
            seen.add("rev")
    assert seen == {"fwd", "rev"}


def test_model_forward_shapes():
    net = RadarOdomNet(GRID, CartesianGridCfg(nx=16, ny=32, nz=8, voxel=0.4), in_channels=4, base=4)
    out = net(torch.randn(2, 4, *GRID.shape))
    assert out.shape == (2, 6)
