import re
import numpy as np

from ase.data import covalent_radii, atomic_numbers
from ions.data import principle_number
from ions.decorate import assign_bvse_params, assign_oxidation_states
from ions.calculators.potential import BVFF

from ..core import Grid
from .._neighborhood import nn_list



def _decompose(mobile_ion):

    """ Decompose input string into chemical element and oxidation state"""

    element = re.sub('\d', '', mobile_ion).replace("+","").replace("-","")
    oxi_state = re.sub('\D', '', mobile_ion)

    if '-' in mobile_ion:
        sign = -1
    else:
        sign = 1
    if len(oxi_state) > 0:
        if sign > 0:
            oxi_state = float(oxi_state)
        else:
            oxi_state = -float(oxi_state)
    else:
        oxi_state = sign

    return element, atomic_numbers[element], int(oxi_state)


def _get_array(atoms, name):
    """ Helper function"""
    arr = atoms.get_array(name)
    arr = np.concatenate([arr, [np.nan]]) # np.nan is added to deal with kDTree upper bound
    return arr



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
    Compute the bond valence site energy (BVSE) distribution for a mobile ion.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure containing the framework and mobile-ion species.
    
    mobile_ion : str
        Chemical symbol of the mobile ion (e.g., "Li", "Na").
    
    resolution : float
        Grid spacing in Å for the energy landscape. Default 0.25.
    
    r_cut : float
        Cutoff radius in Å for bond-valence interactions. Default 10.0.
    
    r_min : float
        Minimum interatomic distance in Å. Default 0.8.
    
    empty_framework : bool
        If True, remove mobile ions before computing the landscape. Default True.
    
    k : int
        Number of bins / grid points along each axis. Default 100.
    
    f : float
        Screening factor (e.g., for the BVSE prefactor). Default 0.74.

    verbose : bool
        print out some info

    Returns
    -------
    gridmlip.Grid, energies
        grid with generated mesh and an array of calculated energies for irreducible mesh points
    """

    symbol, specie, q1 = _decompose(mobile_ion)
    n1, rc1 = principle_number[specie], covalent_radii[specie]

    atoms_decorated = assign_bvse_params(atoms, symbol, q1, self_interaction=True)
    base = atoms_decorated.copy()
    
    if empty_framework:
        base = atoms_decorated[atoms_decorated.numbers != specie]

    grid = Grid(atoms, specie=specie, resolution=resolution,
                r_cut=r_cut, r_min=r_min, empty_framework=empty_framework,
                verbose=verbose)
    grid.generate_mesh()
    mesh_cart = grid.mesh_cart[grid.ir_grid_points]
    mesh_cart = mesh_cart[grid.min_dists > grid.r_min]
    
    distances, _, ids = nn_list(base.positions, mesh_cart, r_cut=r_cut,
                             cell=base.cell, k=k, return_orig_ids=True)

    r_min = np.take(_get_array(base, 'r_min'), ids, axis = -1)
    alpha = np.take(_get_array(base, 'alpha'), ids, axis = -1)
    rc2 = np.take(_get_array(base, 'rc2'), ids, axis = -1)
    d0 = np.take(_get_array(base, 'd0'), ids, axis = -1)
    q2 = np.take(_get_array(base, 'oxi_states'), ids, axis = -1)
    q2 = np.where(q2 * q1 > 0, q2, 0)
    n2 = np.take(_get_array(base, 'n2'), ids, axis = -1)

    morse = np.nan_to_num(BVFF.Morse(distances, r_min, d0, alpha),
                    copy=False,
                    nan=0.0,
            ).sum(axis=1)

    coulomb = np.nan_to_num(BVFF.Coulomb(
                                         q1,
                                         q2,
                                         distances,
                                         rc1,
                                         rc2,
                                         n1,
                                         n2,
                                         f=f),
                    copy=False,
                    nan=0.0,
            ).sum(axis=1)
    
    energies = morse + coulomb
    grid.load_energies(energies)
    
    return grid, energies