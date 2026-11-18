"""Numba-accelerated, displacement-aware periodic percolation on a 3D grid."""
import numpy as np
from numba import njit


@njit(cache=True)
def _find(i, parent, delta):
    """Return root and lattice-cell displacement from root to node i."""
    j = i
    sx = sy = sz = 0
    while parent[j] != j:
        sx += delta[j, 0]
        sy += delta[j, 1]
        sz += delta[j, 2]
        j = parent[j]
    root = j
    j = i
    rx, ry, rz = sx, sy, sz
    while parent[j] != j:
        nxt = parent[j]
        dx, dy, dz = delta[j, 0], delta[j, 1], delta[j, 2]
        parent[j] = root
        delta[j, 0] = rx
        delta[j, 1] = ry
        delta[j, 2] = rz
        rx -= dx
        ry -= dy
        rz -= dz
        j = nxt
    return root, sx, sy, sz


@njit(cache=True)
def _add_winding(root, x, y, z, basis, rank):
    """Add an integer cycle vector if it increases the Q-linear rank."""
    if (x | y | z) == 0:
        return
    r = rank[root]
    if r == 0:
        basis[root, 0, 0] = x
        basis[root, 0, 1] = y
        basis[root, 0, 2] = z
        rank[root] = 1
    elif r == 1:
        a, b, c = basis[root, 0, 0], basis[root, 0, 1], basis[root, 0, 2]
        if (b * z - c * y != 0 or c * x - a * z != 0
                or a * y - b * x != 0):
            basis[root, 1, 0] = x
            basis[root, 1, 1] = y
            basis[root, 1, 2] = z
            rank[root] = 2
    elif r == 2:
        a, b, c = basis[root, 0, 0], basis[root, 0, 1], basis[root, 0, 2]
        d, e, f = basis[root, 1, 0], basis[root, 1, 1], basis[root, 1, 2]
        if ((b * f - c * e) * x + (c * d - a * f) * y
                + (a * e - b * d) * z) != 0:
            basis[root, 2, 0] = x
            basis[root, 2, 1] = y
            basis[root, 2, 2] = z
            rank[root] = 3


@njit(cache=True)
def _join(u, v, wx, wy, wz, parent, size, delta, basis, rank):
    """Add directed u->v edge whose lattice translation is (wx,wy,wz)."""
    a, ax, ay, az = _find(u, parent, delta)
    b, bx, by, bz = _find(v, parent, delta)
    # Lattice translation of root b relative to root a.
    dx = ax + wx - bx
    dy = ay + wy - by
    dz = az + wz - bz
    if a == b:
        _add_winding(a, dx, dy, dz, basis, rank)
        return rank[a]

    if size[a] < size[b]:
        # a becomes child of b; reverse the displacement.
        parent[a] = b
        delta[a, 0] = -dx
        delta[a, 1] = -dy
        delta[a, 2] = -dz
        size[b] += size[a]
        for k in range(rank[a]):
            _add_winding(b, basis[a, k, 0], basis[a, k, 1],
                         basis[a, k, 2], basis, rank)
        return rank[b]
    else:
        parent[b] = a
        delta[b, 0] = dx
        delta[b, 1] = dy
        delta[b, 2] = dz
        size[a] += size[b]
        for k in range(rank[b]):
            _add_winding(a, basis[b, k, 0], basis[b, k, 1],
                         basis[b, k, 2], basis, rank)
        return rank[a]


@njit(cache=True)
def _sweep(values, order, nx, ny, nz, not_found):
    n = values.size
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)
    delta = np.zeros((n, 3), dtype=np.int32)
    basis = np.zeros((n, 3, 3), dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)
    active = np.zeros(n, dtype=np.bool_)
    thresholds = np.full(3, not_found, dtype=np.float64)
    best = 0
    yz = ny * nz

    for t in range(order.size):
        u = order[t]
        active[u] = True
        x = u // yz
        y = (u // nz) % ny
        z = u % nz

        for ox in range(-1, 2):
            px = x + ox
            wx = -1 if px < 0 else (1 if px >= nx else 0)
            px %= nx
            for oy in range(-1, 2):
                py = y + oy
                wy = -1 if py < 0 else (1 if py >= ny else 0)
                py %= ny
                for oz in range(-1, 2):
                    if ox == 0 and oy == 0 and oz == 0:
                        continue
                    pz = z + oz
                    wz = -1 if pz < 0 else (1 if pz >= nz else 0)
                    pz %= nz
                    v = (px * ny + py) * nz + pz
                    if active[v]:
                        r = _join(u, v, wx, wy, wz,
                                  parent, size, delta, basis, rank)
                        if r > best:
                            for d in range(best, r):
                                thresholds[d] = values[u]
                            best = r
                            if best == 3:
                                return thresholds
    return thresholds


class Percolyze:

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float64)
        if self.data.ndim != 3 or min(self.data.shape) == 0:
            raise ValueError("Expected a nonempty 3D array")
        if np.isnan(self.data).any():
            raise ValueError("Data must not contain NaN (use +inf for blocked points)")

    def percolation_barriers(self, upper_limit=10.0):
        if not np.isfinite(upper_limit) or upper_limit <= 0:
            raise ValueError("upper_limit must be finite and positive")

        finite = np.isfinite(self.data)
        if not finite.any():
            return {"e1d": np.inf, "e2d": np.inf, "e3d": np.inf}
        values = (self.data - self.data[finite].min()).ravel()
        eligible = np.flatnonzero(values < upper_limit)
        order = eligible[np.argsort(values[eligible], kind="stable")]
        barriers = _sweep(values, order, *self.data.shape, np.inf)
        return dict(zip(("e1d", "e2d", "e3d"), barriers.tolist()))

    def percolation_radii(self):
        values = np.ascontiguousarray(self.data).ravel()
        order = np.argsort(values, kind="stable")[::-1]
        radii = _sweep(values, order, *self.data.shape, -np.inf)
        return dict(zip(("r1d", "r2d", "r3d"), radii.tolist()))
