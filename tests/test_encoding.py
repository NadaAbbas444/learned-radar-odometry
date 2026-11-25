import numpy as np

from radar_odom.config import CartesianGridCfg, PolarGridCfg
from radar_odom.encoding import encode_numpy, voxel_centres

POLAR = PolarGridCfg(num_range_bins=64, range_bin_width=0.2, range_offset=0.1, num_azimuth_bins=64,
                     num_elevation_bins=16, azimuth_fov_deg=140, elevation_fov_deg=40)
CART = CartesianGridCfg(nx=48, ny=64, nz=16, voxel=0.25)


def splat_point(xyz):
    """Polar image with a single bright bin at the polar coordinates of xyz."""
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    ri = int(np.argmin(abs(POLAR.ranges() - r)))
    ai = int(np.argmin(abs(POLAR.azimuths() - np.arctan2(y, x))))
    ei = int(np.argmin(abs(POLAR.elevations() - np.arctan2(z, np.hypot(x, y)))))
    img = np.zeros((1,) + POLAR.shape, np.float32)
    img[0, ei, ai, ri] = 1.0
    return img


def brightest_voxel_xyz(img):
    c = encode_numpy(img, POLAR, CART)[0]
    return voxel_centres(CART)[np.unravel_index(np.argmax(c), c.shape)]


def test_point_lands_at_its_metric_position():
    # includes the LEFT side (y<0 is right, y>0 left) and BELOW the sensor -- the old
    # encoder only produced the x,y,z >= 0 octant.
    for p in ([5.0, 0.0, 0.0], [4.0, 3.0, 0.5], [4.0, -3.0, -0.8], [8.0, 2.0, 1.0]):
        got = brightest_voxel_xyz(splat_point(np.array(p)))
        # tolerance: angular bin size at that range plus one voxel
        r = np.linalg.norm(p)
        tol = r * np.deg2rad(140 / 63) + CART.voxel
        assert np.linalg.norm(got - p) < tol, (p, got)


def test_outside_fov_is_zero():
    c = encode_numpy(np.ones((1,) + POLAR.shape, np.float32), POLAR, CART)[0]
    xyz = voxel_centres(CART)
    az = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))
    assert c[np.abs(az) > 75].max() == 0
    assert c[np.abs(az) < 60].min() > 0.99 or np.median(c[np.abs(az) < 60]) > 0.99


def test_nonuniform_bins_supported():
    p = PolarGridCfg(num_range_bins=32, range_bin_width=0.3,
                     azimuth_bins=list(np.arcsin(np.linspace(-0.9, 0.9, 32))),
                     elevation_bins=list(np.arcsin(np.linspace(-0.4, 0.4, 8))))
    img = np.random.default_rng(0).random((2,) + p.shape).astype(np.float32)
    out = encode_numpy(img, p, CartesianGridCfg(nx=16, ny=32, nz=8, voxel=0.4))
    assert out.shape == (2, 16, 32, 8) and np.isfinite(out).all()
