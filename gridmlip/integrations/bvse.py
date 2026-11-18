"""Deprecated: use gridmlip.bvse() or Grid(...).evaluate(gridmlip.BVSE(...))."""
import warnings

from ..core import Grid
from ..models.bvse import BVSE, parse_ion


def _decompose(mobile_ion):
    """Decompose input string into chemical element, atomic number and oxidation state."""
    return parse_ion(mobile_ion)


def bvse_distribution(
    atoms: object,
    mobile_ion: str = None,
    resolution: float = 0.25,
    r_cut: float = 10.0,
    r_min: float = 0.8,
    empty_framework: bool = True,
    k: int = 100,
    f: float = 0.74,
    verbose: bool = False,
):
    """
    Deprecated: use ``gridmlip.bvse(atoms, ion)``, which returns an EnergyMap.

    Returns
    -------
    gridmlip.Grid (with energies loaded), energies of the irreducible points
    """
    warnings.warn("gridmlip.integrations.bvse.bvse_distribution is deprecated and will be removed in "
                  "gridmlip 0.4; use gridmlip.bvse(atoms, ion) instead.", DeprecationWarning, stacklevel=2)
    model = BVSE(mobile_ion, cutoff=r_cut, k=k, f=f, verbose=verbose)
    atoms = atoms.copy()
    atoms.set_array("oxi_states", model.resolve_oxidation_states(atoms))
    grid = Grid(atoms, specie=model.z, resolution=resolution, r_cut=r_cut, r_min=r_min,
                empty_framework=empty_framework, verbose=verbose)
    host = model.prepare(grid.atoms, grid._host_mask)
    energies = model(grid.points, host)
    grid.load_energies(energies)
    return grid, energies
