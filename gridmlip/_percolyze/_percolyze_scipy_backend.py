import itertools

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class Percolyze:
    structure = ndi.generate_binary_structure(3, 3)
    shifts = np.array(
        list(itertools.product((0, 1), repeat=3)),
        dtype=np.intp,
    )

    def __init__(self, data):
        data = np.asarray(data)
        if data.ndim != 3 or 0 in data.shape:
            raise ValueError("data must be a nonempty 3-dimensional array")
        if np.isnan(data).any():
            raise ValueError("data must not contain NaN (use +inf for blocked points)")

        self.data = data
        self.size = data.shape
        finite = np.isfinite(data)
        self._energy_data = data - data[finite].min() if finite.any() else np.full(data.shape, np.inf)

    def _dimension_at(self, threshold, *, mode="energy"):

        if mode == "energy":
            mask = np.tile(self._energy_data <= threshold, (2, 2, 2))
        elif mode == "radius":
            mask = np.tile(self.data >= threshold, (2, 2, 2))
        else:
            raise ValueError("mode must be 'energy' or 'radius'")

        labels, n = ndi.label(mask, structure=self.structure)
        if n == 0:
            return 0

        left = (
            labels[-1, :, :],
            labels[:, -1, :],
            labels[:, :, -1],
        )
        right = (
            labels[0, :, :],
            labels[:, 0, :],
            labels[:, :, 0],
        )

        a, b = [], []
        for f1, f2 in zip(left, right):
            valid = (f1 != 0) & (f2 != 0) & (f1 != f2)
            a.append(f1[valid])
            b.append(f2[valid])

        a = np.concatenate(a)
        b = np.concatenate(b)

        if a.size:
            rows = np.concatenate((a, b))
            cols = np.concatenate((b, a))

            graph = coo_matrix(
                (np.ones(rows.size, dtype=np.int32), (rows, cols)),
                shape=(n + 1, n + 1),
            ).tocsr()

            _, group = connected_components(graph, directed=False)
            labels = group[labels]

        features, first = np.unique(labels, return_index=True)
        nonzero = features != 0
        features = features[nonzero]
        first = first[nonzero]

        if features.size == 0:
            return 0

        coords = np.stack(
            np.unravel_index(first, labels.shape),
            axis=-1,
        )

        base = np.asarray(self.size)
        test = (
            (coords % base)[:, None, :]
            + self.shifts[None, :, :] * base
        )

        found = labels[
            test[..., 0],
            test[..., 1],
            test[..., 2],
        ]

        counts = np.count_nonzero(
            found == features[:, None],
            axis=1,
        )
        return int(counts.max())

    def _percolation_energy(self, dim, candidates, cache):
        """Smallest sampled energy at which the region E <= threshold percolates in `dim`.

        Searching over the sorted voxel energies (not a continuous interval) makes the
        result exact and identical to the numba backend.
        """
        lo, hi = 0, len(candidates) - 1
        barrier = np.inf

        while lo <= hi:
            mid = (lo + hi) // 2
            if mid not in cache:
                cache[mid] = self._dimension_at(candidates[mid], mode="energy")

            if cache[mid] >= dim:
                barrier = float(candidates[mid])
                hi = mid - 1  # try a lower energy
            else:
                lo = mid + 1

        return barrier

    def percolation_barriers(self, upper_limit=10.0):
        """Return e1d/e2d/e3d in units above the minimum input energy."""
        if not np.isfinite(upper_limit) or upper_limit <= 0:
            raise ValueError("upper_limit must be finite and positive")
        energies = self._energy_data
        candidates = np.unique(energies[energies < upper_limit])
        cache = {}
        return {
            f"e{i}d": self._percolation_energy(dim, candidates, cache)
            for i, dim in enumerate((2, 4, 8), start=1)
        }

    def _percolation_radius(self, dim, candidates, cache):
        """Find the exact largest radius from sampled voxel clearances."""
        lo, hi = 0, len(candidates) - 1
        radius = -np.inf

        while lo <= hi:
            mid = (lo + hi) // 2
            if mid not in cache:
                cache[mid] = self._dimension_at(
                    candidates[mid], mode="radius"
                )

            if cache[mid] >= dim:
                radius = float(candidates[mid])
                lo = mid + 1  # Try a larger sphere.
            else:
                hi = mid - 1

        return radius

    def percolation_radii(self):
        """Return the largest permeating sphere radii r1d, r2d and r3d."""
        candidates = np.unique(self.data)
        cache = {}
        return {
            f"r{i}d": self._percolation_radius(dim, candidates, cache)
            for i, dim in enumerate((2, 4, 8), start=1)
        }
