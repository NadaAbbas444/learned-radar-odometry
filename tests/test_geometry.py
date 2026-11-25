import numpy as np
from scipy.spatial.transform import Rotation

from radar_odom.geometry import (compose_trajectory, from_twist, interpolate_poses, inv, pose_from_pq,
                                 relative, to_twist, umeyama_se3)


def rand_pose(rng):
    return pose_from_pq(rng.normal(0, 3, 3), Rotation.random(random_state=rng.integers(1e9)).as_quat())


def test_relative_pose_is_invariant_to_world_frame():
    """The bug in the old code: labels must not change if the whole world is rotated/moved."""
    rng = np.random.default_rng(0)
    T1, T2, G = rand_pose(rng), rand_pose(rng), rand_pose(rng)
    a = to_twist(relative(T1, T2))
    b = to_twist(relative(G @ T1, G @ T2))
    np.testing.assert_allclose(a, b, atol=1e-9)
    # the old world-frame difference does change:
    assert np.linalg.norm((G @ T2)[:3, 3] - (G @ T1)[:3, 3] - (T2[:3, 3] - T1[:3, 3])) > 1e-3


def test_relative_pose_known_motion():
    T1 = pose_from_pq([1, 2, 0], Rotation.from_euler("z", 90, degrees=True).as_quat())
    # robot drives 1 m forward (its own x) and turns 10 deg left
    T12 = pose_from_pq([1, 0, 0], Rotation.from_euler("z", 10, degrees=True).as_quat())
    T2 = T1 @ T12
    np.testing.assert_allclose(T2[:3, 3], [1, 3, 0], atol=1e-12)  # world: moved +y
    tw = to_twist(relative(T1, T2))
    np.testing.assert_allclose(tw, [1, 0, 0, 0, 0, np.deg2rad(10)], atol=1e-12)


def test_twist_roundtrip_and_inverse():
    rng = np.random.default_rng(1)
    T = rand_pose(rng)
    np.testing.assert_allclose(from_twist(to_twist(T)), T, atol=1e-9)
    np.testing.assert_allclose(T @ inv(T), np.eye(4), atol=1e-12)


def test_compose_recovers_trajectory():
    rng = np.random.default_rng(2)
    traj = np.stack([rand_pose(rng) for _ in range(20)])
    rel = np.stack([relative(traj[k], traj[k + 1]) for k in range(19)])
    np.testing.assert_allclose(compose_trajectory(rel, traj[0]), traj, atol=1e-9)


def test_interpolation_and_umeyama():
    rng = np.random.default_rng(3)
    T0, T1 = rand_pose(rng), rand_pose(rng)
    mid = interpolate_poses(np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0]), np.stack([T0, T1]))
    np.testing.assert_allclose(mid[0], T0, atol=1e-9)
    np.testing.assert_allclose(mid[1][:3, 3], (T0[:3, 3] + T1[:3, 3]) / 2, atol=1e-9)
    P = rng.normal(size=(50, 3))
    G = rand_pose(rng)
    A = umeyama_se3(P, P @ G[:3, :3].T + G[:3, 3])
    np.testing.assert_allclose(A, G, atol=1e-9)
