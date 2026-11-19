"""The symmetry-reduced mesh must reproduce the brute-force (no symmetry) result exactly."""
import warnings
import numpy as np
import pytest
from spglib import get_symmetry_dataset

from gridmlip import Grid
from gridmlip._mesh import Mesh, commensurate_mesh_shape
from gridmlip._neighborhood import nn_list

from structures import SYMMETRY_CASES, nacl, pseudo_tetragonal


def _symmetrized_vs_exact(atoms, resolution, symprec):
    g = Grid(atoms, specie=3, resolution=resolution, r_cut=6.0, r_min=0.0,
             symprec=symprec, empty_framework=False, verbose=False)
    g.generate_mesh()
    exact, _ = nn_list(g.base.positions, g.mesh_cart, 6.0, g.cell)
    return g, np.abs(g.min_distances.ravel() - exact).max()


@pytest.mark.parametrize("name", SYMMETRY_CASES)
@pytest.mark.parametrize("resolution", [0.2, 0.25, 0.33])
def test_irreducible_mesh_is_exact(name, resolution):
    atoms = SYMMETRY_CASES[name]()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # standard settings: no operation may be dropped
        g, err = _symmetrized_vs_exact(atoms, resolution, symprec=1e-3)
    assert err < 1e-8, f"{name}: symmetrized min-distance differs from exact by {err:.3g} A"

    nops = len(get_symmetry_dataset(
        (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers), symprec=1e-3).rotations)
    assert len(g._mesh.rotations) == nops
    # real compression: at least ~nops/4 reduction (special positions make some orbits smaller)
    assert len(g.ir_grid_points) * nops / 4 <= g.mesh_frac.shape[0]


@pytest.mark.parametrize("name", SYMMETRY_CASES)
def test_mesh_is_periodic_and_commensurate(name):
    atoms = SYMMETRY_CASES[name]()
    mesh = Mesh.from_atoms(atoms, resolution=0.25, symprec=1e-3)
    n = np.array(mesh.mesh_shape)
    frac = mesh.frac_coords
    assert frac.min() == 0 and frac.max() < 1  # no repeated boundary plane
    lengths = np.linalg.norm(atoms.cell[:], axis=1)
    # same density rule as before (n = L // resolution), only ever rounded up
    assert np.all(n >= lengths // 0.25)
    for R, t in zip(mesh.rotations, mesh.translations):
        assert np.all(t == np.rint(t))
        assert not np.any((R != 0) & (n[:, None] != n[None, :]))


def test_pseudo_symmetric_cell_is_bounded_by_symprec():
    """a=4.00, b=4.04 is 'tetragonal' at symprec=0.1. Old version: 0.57 A error."""
    _, err = _symmetrized_vs_exact(pseudo_tetragonal(), 0.2, symprec=0.1)
    assert err < 0.05


def test_commensurate_shape_rules():
    hexagonal_3fold = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]])
    from fractions import Fraction
    shape = commensurate_mesh_shape(
        (24, 25, 26),
        [np.eye(3, dtype=int), hexagonal_3fold],
        [[Fraction(0)] * 3, [Fraction(0), Fraction(0), Fraction(1, 3)]],
    )
    assert shape[0] == shape[1] == 25
    assert shape[2] % 3 == 0 and shape[2] >= 26


def test_incommensurate_ops_are_dropped_not_misapplied():
    """An explicit mesh that does not fit the ops: ops are dropped with a warning, result stays exact."""
    R = [np.eye(3, dtype=int), np.eye(3, dtype=int)]
    t = [np.zeros(3), np.array([0, 0, 1 / 3])]
    with pytest.warns(UserWarning, match="not commensurate"):
        mesh = Mesh((4, 4, 10), R, t)
    assert len(mesh.rotations) == 1
    ir, mapping = mesh.get_irreducible_mapping()
    assert len(ir) == 4 * 4 * 10


def test_oxidation_states_lower_symmetry():
    """Charge ordering of the same element must not be symmetrized away."""
    atoms = nacl()
    plain = Mesh.from_atoms(atoms, resolution=0.3, symprec=1e-3)
    oxi = atoms.get_array("oxi_states").copy()
    oxi[np.flatnonzero(atoms.numbers == 11)[0]] = 2
    atoms.set_array("oxi_states", oxi)
    ordered = Mesh.from_atoms(atoms, resolution=0.3, symprec=1e-3)
    assert len(ordered.rotations) < len(plain.rotations)


def test_bvse_charge_ordered_matches_no_symmetry():
    """Charges set before the grid: the grid symmetry itself respects the charge ordering."""
    pytest.importorskip("ions")
    from gridmlip import BVSE

    atoms = nacl()
    oxi = atoms.get_array("oxi_states").copy()
    oxi[np.flatnonzero(atoms.numbers == 11)[0]] = 2  # one Na2+ breaks the cubic symmetry
    atoms.set_array("oxi_states", oxi)
    # 5.64/0.2349 -> 24 points (commensurate); k large enough that the cutoff, not k, truncates
    kw = dict(specie=3, resolution=0.2349, r_cut=8.0, r_min=0.8)
    model = BVSE("Li+", cutoff=8.0, k=400)

    sym = Grid(atoms, **kw)
    nosym = Grid(atoms, symprec=None, **kw)
    a, b = sym.evaluate(model), nosym.evaluate(model)

    assert sym.mesh_shape == nosym.mesh_shape
    assert len(sym.ir_grid_points) < len(nosym.ir_grid_points)
    np.testing.assert_allclose(a.values, b.values, rtol=0, atol=1e-10)
