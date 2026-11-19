"""The v0.3 public API: Grid -> evaluate / load_energies -> EnergyMap."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

import gridmlip as gm
from gridmlip import BVSE, EnergyMap, Grid
from gridmlip.predictors import CalculatorPredictor, as_predictor

from structures import lifepo4, nacl


# ---------------------------------------------------------------- helpers

def probe_distance_energy(atoms_list):
    """Toy predictor: energy = -(distance from the probe (last atom) to its nearest neighbor)."""
    out = []
    for a in atoms_list:
        d = a.get_distances(len(a) - 1, range(len(a) - 1), mic=True)
        out.append(-d.min())
    return np.array(out)


@pytest.fixture
def al_grid():
    """fcc Al host, H probe: small enough for EMT one structure at a time."""
    host = bulk("Al", "fcc", a=4.05, cubic=True)
    return Grid(host, specie="H", resolution=0.5, r_cut=3.0, r_min=1.2)


# ------------------------------------------------------------------- Grid

def test_specie_accepts_symbol_and_number():
    assert Grid(nacl(), "Na").specie == Grid(nacl(), 11).specie == 11


def test_points_and_configurations_are_aligned(al_grid):
    pts = al_grid.points
    cfgs = al_grid.prepare_configurations()
    assert len(cfgs) == len(al_grid) == len(pts)
    for i, c in enumerate(cfgs):
        assert c.info["grid_index"] == i
        assert c.numbers[-1] == al_grid.specie
        np.testing.assert_allclose(c.positions[-1], pts[i])


def test_supercell_cutoff(al_grid):
    small = al_grid.prepare_configurations(supercell_cutoff=3.0)[0]
    large = al_grid.prepare_configurations(supercell_cutoff=8.0)[0]
    assert len(large) > len(small)


def test_evaluate_callable_equals_load_energies(al_grid):
    a = al_grid.evaluate(probe_distance_energy)
    b = al_grid.load_energies(probe_distance_energy(al_grid.prepare_configurations()))
    np.testing.assert_array_equal(a.values, b.values)
    assert isinstance(a, EnergyMap) and a.values.shape == al_grid.mesh_shape


def test_evaluate_ase_calculator(al_grid):
    assert isinstance(as_predictor(EMT()), CalculatorPredictor)
    emap = al_grid.evaluate(EMT())
    by_hand = []
    for c in al_grid.prepare_configurations():
        c.calc = EMT()
        by_hand.append(c.get_potential_energy())
    np.testing.assert_allclose(emap.values, al_grid.load_energies(by_hand).values)
    assert "EMT" in emap.info["model"]
    assert np.isfinite(emap.percolation_barriers()["e3d"])


def test_evaluate_rejects_non_models(al_grid):
    with pytest.raises(TypeError, match="Cannot evaluate"):
        al_grid.evaluate(42)


def test_load_energies_from_file(al_grid, tmp_path):
    cfgs = al_grid.prepare_configurations()
    energies = probe_distance_energy(cfgs)
    for c, e in zip(cfgs, energies):
        c.calc = SinglePointCalculator(c, energy=e)
    write(tmp_path / "processed.xyz", cfgs)
    emap = al_grid.load_energies(tmp_path / "processed.xyz")
    np.testing.assert_allclose(emap.values, al_grid.load_energies(energies).values)
    assert emap.info["source"].endswith("processed.xyz")


# -------------------------------------------------------------- EnergyMap

def test_energy_map_save_load_write(al_grid, tmp_path):
    emap = al_grid.evaluate(probe_distance_energy)
    emap.save(tmp_path / "m.npz")
    back = EnergyMap.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(back.values, emap.values)
    assert back.specie == emap.specie and back.info == emap.info
    assert back.percolation_barriers() == emap.percolation_barriers()

    emap.write(tmp_path / "m.grd")
    emap.write(tmp_path / "m.cube")
    assert (tmp_path / "m.grd").stat().st_size > 0 and (tmp_path / "m.cube").stat().st_size > 0
    with pytest.raises(ValueError, match="extension"):
        emap.write(tmp_path / "m.txt")


def test_local_minima_finds_octahedral_sites():
    atoms = nacl()
    grid = Grid(atoms, "Na", resolution=0.2, r_min=0.5)
    # energy = -distance to the host: minima are the largest holes (the Na sites)
    emap = grid.evaluate(probe_distance_energy)
    sites = emap.local_minima(min_distance=2.0, energy_window=0.2, coord_type="frac")
    assert len(sites) == 4
    na = atoms.get_scaled_positions()[atoms.numbers == 11]
    for frac, _, _ in sites:
        d = (na - frac + 0.5) % 1 - 0.5
        assert np.min(np.linalg.norm(d @ atoms.cell[:], axis=1)) < 0.2


# ------------------------------------------------------------------- BVSE

@pytest.fixture
def need_ions():
    pytest.importorskip("ions")


def test_bvse_shortcut_equals_explicit_grid(need_ions):
    atoms = lifepo4()
    a = gm.bvse(atoms, "Li+", resolution=0.3)
    b = Grid(atoms, "Li", resolution=0.3, r_cut=10.0, r_min=0.8).evaluate(BVSE("Li+"))
    np.testing.assert_allclose(a.values, b.values, rtol=0, atol=1e-10)
    assert a.percolation_barriers() == b.percolation_barriers()


def test_bvse_uses_existing_charges(need_ions):
    atoms = nacl()
    atoms.set_array("oxi_states", np.where(atoms.numbers == 11, 1, -1))
    emap = gm.bvse(atoms, "Na+", resolution=0.3, verbose=False)
    assert emap.info["oxidation_states_source"] == "atoms.arrays['oxi_states']"


def test_bvse_assigns_missing_charges_with_pymatgen(need_ions, capsys):
    pytest.importorskip("pymatgen")
    atoms = lifepo4()
    del atoms.arrays["oxi_states"]
    emap = gm.bvse(atoms, "Li+", resolution=0.3)
    assert emap.info["oxidation_states_source"] == "pymatgen BVAnalyzer"
    assert "Fe+2" in capsys.readouterr().out
    ref = gm.bvse(lifepo4(), "Li+", resolution=0.3, verbose=False)  # same charges, set by hand
    np.testing.assert_allclose(emap.values, ref.values, rtol=0, atol=1e-10)


def test_bvse_charge_ordering_resolved_inside_evaluate(need_ions):
    """Grid built without charges (full cubic symmetry); BVSE brings ordered charges:
    evaluate must drop the symmetry operations that mix Na+ and Na2+."""
    atoms = nacl()
    del atoms.arrays["oxi_states"]
    charges = np.where(atoms.numbers == 11, 1, -1)
    charges[np.flatnonzero(atoms.numbers == 11)[0]] = 2
    model = BVSE("Li+", cutoff=8.0, k=400, oxidation_states=charges)
    kw = dict(specie=3, resolution=0.2349, r_cut=8.0, r_min=0.8)

    a = Grid(atoms, **kw).evaluate(model)
    b = Grid(atoms, symprec=None, **kw).evaluate(model)
    np.testing.assert_allclose(a.values, b.values, rtol=0, atol=1e-10)


def test_bvse_oxidation_states_dict(need_ions):
    emap = gm.bvse(nacl(), "Na+", resolution=0.3, oxidation_states={"Na": 1, "Cl": -1})
    assert emap.info["oxidation_states_source"].startswith("user")


# ---------------------------------------------------------------- defects

def test_defect_functions(need_ions):
    from gridmlip.defects import frenkel_pairs, interstitials, vacancies

    atoms = lifepo4()
    kw = dict(resolution=0.4, r_min=1.3, r_cut=6.0)
    for func, name in [(interstitials, "interstitial"), (vacancies, "vacancy"),
                       (frenkel_pairs, "frenkel_pair")]:
        cfgs = func(atoms, 3, **kw)
        assert func(atoms, 3, dry_run=True, **kw)["points"] == len(cfgs) > 0
        assert all(c.info["defect"] == name for c in cfgs)


def test_deprecated_names_are_gone():
    assert not hasattr(Grid, "construct_cfgs") and not hasattr(Grid, "write_grd")
    with pytest.raises(ImportError):
        import gridmlip.integrations  # noqa: F401
    import gridmlip.defects
    assert not hasattr(gridmlip.defects, "DefectGenerator")
