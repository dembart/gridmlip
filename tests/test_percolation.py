"""Both percolation backends must give identical, exact answers."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from gridmlip._percolyze._percolyze_scipy_backend import Percolyze as ScipyPercolyze

numba_backend = pytest.importorskip("gridmlip._percolyze._percolyze_numba_backend")
NumbaPercolyze = numba_backend.Percolyze


def _fields():
    rng = np.random.default_rng(0)
    out = []
    for shape, sigma in [((12, 14, 10), 1.5), ((20, 16, 18), 2.5), ((9, 9, 30), (1, 1, 4))]:
        f = gaussian_filter(rng.normal(size=shape), sigma, mode="wrap")
        out.append(3 * (f - f.min()) / np.ptp(f))
    blocked = out[1].copy()
    blocked[blocked > 2.0] = np.inf
    out.append(blocked)
    channel = np.full((10, 10, 10), 5.0)  # 1D channel along x with a 0.7 bottleneck
    channel[:, 5, 5] = 0.0
    channel[3, 5, 5] = 0.7
    out.append(channel)
    return out


@pytest.mark.parametrize("i", range(5))
def test_backends_agree_on_barriers(i):
    field = _fields()[i]
    a = NumbaPercolyze(field).percolation_barriers(upper_limit=10.0)
    b = ScipyPercolyze(field).percolation_barriers(upper_limit=10.0)
    assert a == b


@pytest.mark.parametrize("i", range(4))
def test_backends_agree_on_radii(i):
    field = np.where(np.isfinite(_fields()[i]), _fields()[i], 0.0)
    assert NumbaPercolyze(field).percolation_radii() == ScipyPercolyze(field).percolation_radii()


def test_known_channel():
    field = _fields()[4]
    for impl in (NumbaPercolyze, ScipyPercolyze):
        r = impl(field).percolation_barriers(upper_limit=4.0)  # bulk (5.0) is above the limit
        assert r["e1d"] == pytest.approx(0.7)
        assert r["e2d"] == np.inf and r["e3d"] == np.inf


def test_upper_limit():
    field = _fields()[4]
    for impl in (NumbaPercolyze, ScipyPercolyze):
        assert impl(field).percolation_barriers(upper_limit=0.5)["e1d"] == np.inf


def test_nan_rejected():
    field = _fields()[0].copy()
    field[0, 0, 0] = np.nan
    for impl in (NumbaPercolyze, ScipyPercolyze):
        with pytest.raises(ValueError):
            impl(field)
