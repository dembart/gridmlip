import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from gridmlip import BVSE, EnergyMap, Grid, Hop, Hops, HopMaps
from ions import Percolator

from structures import nacl, lifepo4


def _toy_map():
    """Two wells along x on a high background, a bump on the line between them and a channel around it."""
    L, n = 10.0, 50
    x = np.arange(n) / n * L
    r = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    p0, p1, c = np.array([3.0, 5.0, 5.0]), np.array([7.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])
    g = lambda p, w: np.exp(-np.sum((r - p) ** 2, -1) / w)
    values = 1.0 - 1.5 * g(p0, 0.5) - 1.5 * g(p1, 0.5) + 0.8 * g(c, 0.5) - 0.9 * g(c + [0, 1.6, 0], 0.6)
    return EnergyMap(values, Atoms(cell=[L, L, L], pbc=True)), p0, p1


@pytest.fixture(scope="module")
def lfp_hops():
    atoms = lifepo4()
    pl = Percolator(atoms, "Li", upper_bound=6.3)
    cutoff = pl.cutoff_search(1, 0.5)
    return Hops(atoms, "Li", pl.unique_edges(cutoff, 0.5), resolution=0.3)


# ------------------------------------------------------------------ region

def test_region_restricts_evaluation():
    g = Grid(nacl(), specie=11, resolution=0.3, r_cut=5.0, r_min=1.0, verbose=False)
    model = BVSE("Na+", verbose=False)
    full = g.evaluate(model)
    center = np.diag(g.atoms.cell).sum() / 3 * np.array([0.25, 0.25, 0.25])
    region = lambda pts: np.linalg.norm(pts - center, axis=1) < 2.0
    part = g.evaluate(model, region=region)
    inside = np.isfinite(part.values)
    assert 0 < part.info["grid"]["n_points"] < full.info["grid"]["n_points"]
    assert part.info["grid"]["region"]
    # evaluated points have the same energies as in the full map
    assert np.allclose(part.values[inside], full.values[inside])


def test_region_configurations_and_load_agree():
    g = Grid(nacl(), specie=11, resolution=0.3, r_cut=5.0, r_min=1.0, verbose=False)
    region = np.zeros(g.mesh_shape, dtype=bool)
    region[:3] = True
    cfgs = g.prepare_configurations(region=region)
    assert len(cfgs) == len(g.region_points(region)) < len(g)
    emap = g.load(np.arange(len(cfgs), dtype=float), region=region)
    assert np.isfinite(emap.values).sum() >= len(cfgs)
    with pytest.raises(ValueError, match="region must cover"):
        g.region_points(np.ones(5, dtype=bool))


# ------------------------------------------------------------------ paths

def test_path_goes_around_a_bump():
    emap, p0, p1 = _toy_map()
    path = emap.path(p0, p1, n_images=12)
    assert path.points.shape == (12, 3)
    assert np.allclose(path.points[0], p0) and np.allclose(path.points[-1], p1)
    assert path.level < emap.values[15:36, 25, 25].max() - 0.3   # lower than the straight line
    assert np.linalg.norm(np.cross(path.saddle - p0, p1 - p0)) / np.linalg.norm(p1 - p0) > 0.8
    steps = np.linalg.norm(np.diff(path.points, axis=0), axis=1)
    assert np.allclose(steps, steps.mean(), rtol=0.05)
    with pytest.raises(ValueError, match="needs a hop"):
        path.images()


def test_path_crosses_the_periodic_boundary():
    emap, p0, p1 = _toy_map()
    shift = np.array([4.5, 0.0, 0.0])                        # wells at x = 7.5 and 11.5 (= 1.5 wrapped)
    rolled = EnergyMap(np.roll(emap.values, 23, axis=0), emap.atoms)
    path = rolled.path(p0 + shift + [0.1, 0, 0], p1 + shift + [0.1, 0, 0], n_images=8)
    assert path.points[-1, 0] > 10.0                          # unwrapped, not folded back into the cell
    assert np.all(np.diff(path.points[:, 0]) > -0.5)


def test_path_rejects_displacement_longer_than_half_cell():
    emap, p0, _ = _toy_map()
    with pytest.raises(ValueError, match="half the cell"):
        emap.path(p0, p0 + [6.0, 0, 0])


# ------------------------------------------------------------------ hops

def test_hop_frame():
    atoms = lifepo4()
    pl = Percolator(atoms, "Li", upper_bound=6.3)
    s, t, *o = pl.unique_edges(pl.cutoff_search(1, 0.5), 0.5)[0]
    hop = Hop(atoms, s, t, o)
    n = len(atoms) * int(np.prod(hop.scale))
    assert len(hop.supercell) == n and len(hop.template) == n - 1 and len(hop.host) == n - 2
    assert hop.template.get_array("moving").sum() == 1
    assert hop.template[hop.moving].symbol == "Li"
    assert hop.length == pytest.approx(3.0, abs=0.05)
    with pytest.raises(ValueError, match="different species"):
        Hop(atoms, s, int(np.flatnonzero(atoms.symbols == "Fe")[0]), o)


def test_hops_keep_order_and_one_supercell(lfp_hops):
    assert len(lfp_hops) == len(lfp_hops.edges)
    for hop, (s, t, *o) in zip(lfp_hops, lfp_hops.edges):
        assert (hop.source, hop.target, hop.offset) == (s, t, tuple(o))
        assert hop.scale == lfp_hops.scale


def test_hop_paths_and_images(lfp_hops):
    paths = lfp_hops.evaluate(BVSE("Li+", verbose=False)).paths(n_images=7)
    assert len(paths) == len(lfp_hops)
    images = paths.images()[0]
    hop = lfp_hops[0]
    assert len(images) == 7
    assert all(len(im) == len(hop.template) and im.get_array("moving").sum() == 1 for im in images)
    assert np.allclose(images[0].positions[hop.moving], hop.p0)
    assert np.allclose(images[-1].positions[hop.moving], hop.p1)
    assert np.all(np.isfinite(paths.barriers)) and np.all(paths.barriers > 0)


def test_hops_external_roundtrip(lfp_hops):
    predictor = lambda cfgs: np.array([cfg.positions[-1, 0] for cfg in cfgs])   # any callable list[Atoms] -> energies
    cfgs = lfp_hops.prepare_configurations()
    assert [c.info["hop_index"] for c in cfgs] == sorted(c.info["hop_index"] for c in cfgs)
    loaded = lfp_hops.load(predictor(cfgs))
    direct = lfp_hops.evaluate(predictor)
    for a, b in zip(loaded, direct):
        assert np.array_equal(np.isfinite(a.values), np.isfinite(b.values))
        assert np.allclose(a.values[np.isfinite(a.values)], b.values[np.isfinite(b.values)])
    with pytest.raises(ValueError, match="Expected"):
        lfp_hops.load(np.zeros(3))


def test_maps_save_load_and_paths_write(lfp_hops, tmp_path):
    maps = lfp_hops.evaluate(BVSE("Li+", verbose=False))
    maps.save(tmp_path / "maps")
    again = HopMaps.load(tmp_path / "maps", lfp_hops)
    assert np.array_equal(again[0].values, maps[0].values)

    paths = again.paths(n_images=5)
    paths.write(tmp_path / "xyz")
    assert len(read(tmp_path / "xyz" / "hop_000.xyz", ":")) == 5
    summary = json.loads((tmp_path / "xyz" / "paths.json").read_text())
    assert summary[0]["barrier"] == pytest.approx(paths.barriers[0])

    paths.write(tmp_path / "vasp", format="vasp")
    assert (tmp_path / "vasp" / "hop_000" / "04" / "POSCAR").exists()


# ------------------------------------------------------------------ BVSE oxidation states

@pytest.fixture
def unit_cell_only_assignment(monkeypatch):
    """Oxidation states assigned only on the unit cell: like pymatgen on a cell with a vacancy, any
    other cell fails. Returns the atoms (no oxi_states) and the list of assigned cell sizes."""
    import importlib

    bvse = importlib.import_module("gridmlip.models.bvse")   # the module (gridmlip.models.bvse is also a function)

    atoms = lifepo4()
    q = atoms.get_array("oxi_states").copy()
    del atoms.arrays["oxi_states"]
    calls = []

    def assign(a, *args, **kwargs):
        calls.append(len(a))
        if len(a) != len(atoms):
            raise ValueError("Valences cannot be assigned!")
        return q.copy()

    monkeypatch.setattr(bvse, "assign_oxidation_states", assign)
    return atoms, q, calls


def test_hop_assigns_oxidation_states_on_the_unit_cell(unit_cell_only_assignment):
    atoms, q, calls = unit_cell_only_assignment
    pl = Percolator(atoms, "Li", upper_bound=6.3)
    s, t, *o = pl.unique_edges(pl.cutoff_search(1, 0.5), 0.5)[0]
    hop = Hop(atoms, s, t, o)
    path = hop.evaluate(BVSE("Li+", verbose=False), resolution=0.3).path(n_images=5)
    assert calls == [len(atoms)] and path.barrier > 0
    reps = int(np.prod(hop.scale))
    assert np.array_equal(hop.supercell.get_array("oxi_states"), np.tile(q, reps))
    assert hop.host.get_array("oxi_states").sum() == np.tile(q, reps).sum() - 2           # two Li+ fewer
    assert "oxi_states" not in atoms.arrays                                                  # caller's atoms untouched


def test_hops_assign_oxidation_states_once(unit_cell_only_assignment):
    atoms, q, calls = unit_cell_only_assignment
    pl = Percolator(atoms, "Li", upper_bound=6.3)
    hops = Hops(atoms, "Li", pl.unique_edges(pl.cutoff_search(1, 0.5), 0.5), resolution=0.3)
    maps = hops.evaluate(BVSE("Li+", verbose=False))
    assert calls == [len(atoms)] and len(maps) == len(hops)
    assert all(np.array_equal(h.atoms.get_array("oxi_states"), q) for h in hops)


def test_bvse_per_element_oxidation_states_need_no_assignment(unit_cell_only_assignment):
    atoms, q, calls = unit_cell_only_assignment
    pl = Percolator(atoms, "Li", upper_bound=6.3)
    s, t, *o = pl.unique_edges(pl.cutoff_search(1, 0.5), 0.5)[0]
    Hop(atoms, s, t, o).evaluate(BVSE("Li+", oxidation_states={"Li": 1, "Fe": 2, "P": 5, "O": -2}, verbose=False),
                                 resolution=0.3)
    assert calls == []
