"""Test structures built in code (no data files needed)."""
import numpy as np
from ase import Atoms
from ase.spacegroup import crystal


def _with_oxi(atoms, oxi):
    atoms.set_array("oxi_states", np.array([oxi[s] for s in atoms.get_chemical_symbols()], dtype=int))
    return atoms


def nacl():
    at = crystal(["Na", "Cl"], basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
                 spacegroup=225, cellpar=[5.64, 5.64, 5.64, 90, 90, 90])
    return _with_oxi(at, {"Na": 1, "Cl": -1})


def lifepo4():
    at = crystal(["Li", "Fe", "P", "O", "O", "O"],
                 basis=[(0, 0, 0), (0.282, 0.25, 0.975), (0.095, 0.25, 0.418),
                        (0.097, 0.25, 0.743), (0.457, 0.25, 0.206), (0.166, 0.046, 0.285)],
                 spacegroup=62, cellpar=[10.33, 6.01, 4.69, 90, 90, 90])
    return _with_oxi(at, {"Li": 1, "Fe": 2, "P": 5, "O": -2})


def licoo2():
    at = crystal(["Li", "Co", "O"], basis=[(0, 0, 0.5), (0, 0, 0), (0, 0, 0.26)],
                 spacegroup=166, cellpar=[2.82, 2.82, 14.05, 90, 90, 120])
    return _with_oxi(at, {"Li": 1, "Co": 3, "O": -2})


def limn2o4():
    at = crystal(["Li", "Mn", "O"],
                 basis=[(0.125, 0.125, 0.125), (0.5, 0.5, 0.5), (0.263, 0.263, 0.263)],
                 spacegroup=227, setting=2, cellpar=[8.24, 8.24, 8.24, 90, 90, 90])
    return _with_oxi(at, {"Li": 1, "Mn": 4, "O": -2})


def quartz():
    """P3_221: 1/3 screw translations along c."""
    return crystal(["Si", "O"], basis=[(0.4697, 0, 0), (0.4135, 0.2669, 0.1191)],
                   spacegroup=154, cellpar=[4.913, 4.913, 5.405, 90, 90, 120])


def p61():
    """P6_1: 1/6 screw translations along c."""
    return crystal(["Li", "O"], basis=[(0.1, 0.2, 0.05), (0.35, 0.1, 0.3)],
                   spacegroup=169, cellpar=[6.0, 6.0, 9.0, 90, 90, 120])


def p41():
    """P4_1: 1/4 screw translations along c."""
    return crystal(["Li", "O"], basis=[(0.1, 0.2, 0.05), (0.3, 0.15, 0.3)],
                   spacegroup=76, cellpar=[5.0, 5.0, 7.3, 90, 90, 90])


def pseudo_tetragonal():
    """Tetragonal within symprec=0.1 but a != b (4.00 vs 4.04 A)."""
    return Atoms("LiO", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
                 cell=[4.0, 4.04, 6.0], pbc=True)


# name -> (builder, mobile ion for BVSE or None)
SYMMETRY_CASES = {
    "nacl_Fm-3m": nacl,
    "lifepo4_Pnma": lifepo4,
    "licoo2_R-3m": licoo2,
    "limn2o4_Fd-3m": limn2o4,
    "quartz_P3221": quartz,
    "p61_P6_1": p61,
    "p41_P4_1": p41,
}

BVSE_CASES = {
    "nacl": (nacl, "Na+"),
    "lifepo4": (lifepo4, "Li+"),
    "licoo2": (licoo2, "Li+"),
    "limn2o4": (limn2o4, "Li+"),
}
