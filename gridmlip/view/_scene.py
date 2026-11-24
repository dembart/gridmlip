"""Data handed to the page: raw frames, volumetric grids and the element tables (the page builds the scene)."""
import base64

import numpy as np
from ase.data import chemical_symbols, covalent_radii

from ._vesta import VESTA, VESTA_POLYHEDRA


def is_periodic(atoms):
    return bool(np.any(atoms.pbc) and atoms.cell.volume > 1e-6)


def tables():
    """Element tables for the page: VESTA colors/radii and polyhedra rules, covalent radii, Z -> symbol."""
    return {
        "colors": {e: [c, r] for e, (c, r) in VESTA.items()},
        "rules": VESTA_POLYHEDRA,
        "covalent": {s: round(float(r), 3) for s, r in zip(chemical_symbols, covalent_radii) if s != "X"},
        "symbols": list(chemical_symbols),
    }


SPGLIB_MAX_ATOMS = 2000  # spglib time grows ~N²: 3.6 s at 14k atoms, ~5 min at 115k; big cells rarely need it


def spacegroup(atoms, symprec=0.01):
    """'Pnma (62)' from spglib when it is installed (optional) and the cell is not too big, else None."""
    if not is_periodic(atoms) or len(atoms) == 0 or len(atoms) > SPGLIB_MAX_ATOMS:
        return None
    try:
        import spglib
    except ImportError:
        return None
    try:
        return spglib.get_spacegroup((np.asarray(atoms.cell), atoms.get_scaled_positions(), atoms.numbers), symprec=symprec)
    except Exception:
        return None


def frame_payload(atoms, data=None, labels=None, name="", symmetry=True):
    periodic = is_periodic(atoms)
    return {
        "spacegroup": spacegroup(atoms) if symmetry else None,
        "symbols": atoms.get_chemical_symbols(),
        "xyz": np.round(atoms.positions, 5).tolist(),
        "cell": np.round(np.asarray(atoms.cell), 6).tolist() if periodic else None,
        "periodic": periodic,
        "data": data or {},
        "labels": labels,
        "name": name,
    }


def encode_grid(values, limit):
    """Grid -> (shape, base64 little-endian float32 in C order), resampled periodically to at most `limit` points."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("volume has no finite values")
    values = np.where(finite, values, values[finite].max())  # e.g. blocked points of an energy map
    if limit and values.size > limit:
        from scipy.ndimage import map_coordinates

        f = (limit / values.size) ** (1 / 3)
        shape = [max(2, int(round(n * f))) for n in values.shape]
        # new point k sits at fractional k / m, i.e. at old index k * n / m (grid point i is at i / n)
        axes = [np.arange(m) * n / m for m, n in zip(shape, values.shape)]
        values = map_coordinates(values, np.meshgrid(*axes, indexing="ij"), order=1, mode="grid-wrap")
    data = values.astype("<f4")
    return list(data.shape), base64.b64encode(data.tobytes()).decode("ascii")
