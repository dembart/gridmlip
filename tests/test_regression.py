"""
Compare with the previous version (v0.2.7, run without symmetry, see make_reference.py).

The mesh changed (periodic i/n instead of i/(n-1) with a repeated boundary plane, and sizes
rounded up to be commensurate with symmetry), so values are compared within a discretization
tolerance, not bit for bit.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from gridmlip import Grid

from structures import BVSE_CASES, SYMMETRY_CASES

REF = json.loads((Path(__file__).parent / "data" / "reference_v0_2_7.json").read_text())

E_TOL = 0.05   # eV, BVSE barriers on a 0.2 A mesh
R_TOL = 0.05   # A, percolation radii on a 0.2 A mesh


def _close(new, old, tol):
    if np.isinf(old) or np.isinf(new):
        return np.isinf(old) and np.isinf(new)
    return abs(new - old) <= tol


@pytest.mark.parametrize("name", BVSE_CASES)
def test_bvse_barriers_match_previous_version(name):
    pytest.importorskip("ions")
    import gridmlip as gm

    build, ion = BVSE_CASES[name]
    p = REF["params"]
    emap = gm.bvse(build(), ion, resolution=p["resolution"], r_min=p["r_min"], cutoff=p["r_cut"],
                   k=p["k"], f=p["f"], empty_framework=True)
    new = emap.percolation_barriers()
    old = REF["bvse"][name]["barriers"]
    for key in ("e1d", "e2d", "e3d"):
        assert _close(new[key], old[key], E_TOL), f"{name} {key}: new {new[key]:.4f} vs old {old[key]:.4f}"


@pytest.mark.parametrize("name", SYMMETRY_CASES)
def test_percolation_radii_match_previous_version(name):
    g = Grid(SYMMETRY_CASES[name](), specie=3, resolution=0.2, r_cut=6.0, r_min=0.5,
             symprec=0.1, empty_framework=True, verbose=False)
    new = g.percolation_radii()
    old = REF["radii"][name]
    for key in ("r1d", "r2d", "r3d"):
        assert _close(new[key], old[key], R_TOL), f"{name} {key}: new {new[key]:.4f} vs old {old[key]:.4f}"
