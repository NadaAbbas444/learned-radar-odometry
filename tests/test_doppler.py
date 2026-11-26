import numpy as np
import torch

from radar_odom.doppler import DopplerVelocityNet, lsq_velocity


def static_scene_points(v, w, n=400, seed=0):
    """Doppler of static reflectors seen by a radar moving with linear v, angular w."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3)); u[:, 0] = np.abs(u[:, 0]); u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rng.uniform(1, 7, n)
    p = u * r[:, None]
    d = np.einsum("ij,ij->i", u, -(v + np.cross(w, p)))
    return np.concatenate([u, r[:, None], d[:, None], np.ones((n, 1))], 1).astype(np.float32)


def test_doppler_recovers_velocity_and_is_blind_to_rotation():
    v = np.array([0.8, -0.3, 0.1])
    a = static_scene_points(v, np.zeros(3))
    b = static_scene_points(v, np.array([0.0, 0.2, 1.5]))  # same v, fast rotation
    np.testing.assert_allclose(a[:, 4], b[:, 4], atol=1e-6)  # identical Doppler: rotation unobservable
    np.testing.assert_allclose(lsq_velocity(a, keep_frac=1.0, ridge=0), v, atol=1e-5)


def test_doppler_net_shapes_and_physics_solve():
    net = DopplerVelocityNet()
    p = torch.from_numpy(np.stack([static_scene_points(np.array([0.5, 0.2, 0.0]), np.zeros(3), seed=i)
                                   for i in range(4)]))
    v, w = net(p)
    assert v.shape == (4, 3) and w.shape == (4, 400)
    # any positive weights solve the noise-free system exactly (up to the small ridge)
    np.testing.assert_allclose(v.detach().numpy(), np.tile([0.5, 0.2, 0.0], (4, 1)), atol=0.02)
