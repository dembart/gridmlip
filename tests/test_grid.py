import numpy as np
import pytest

from gridmlip import Grid
from gridmlip._void_search import clustered_local_minima

from structures import nacl


@pytest.fixture
def blocked_grid():
    g = Grid(nacl(), specie=11, resolution=0.3, r_cut=5.0, r_min=1.5, verbose=False)
    g._generate_mesh()
    accessible = g._min_dists > g.r_min
    assert (~accessible).any()
    g.emap = g.load(-g._min_dists[accessible])
    return g


def test_blocked_points_are_inf(blocked_grid):
    g = blocked_grid
    blocked = (g._min_dists <= g.r_min)[g._mapping].reshape(g.mesh_shape)
    v = g.emap.values
    assert np.isinf(v[blocked]).all()
    assert np.isfinite(v[~blocked]).all()
    assert np.nanmax(np.abs(v[~blocked])) < 1e3


def test_percolation_with_blocked_points(blocked_grid):
    for backend in ("numba", "scipy"):
        pytest.importorskip("numba") if backend == "numba" else None
        r = blocked_grid.emap.percolation_barriers(backend=backend)
        assert np.isfinite(r["e3d"])


def test_nan_energy_becomes_blocked():
    g = Grid(nacl(), specie=11, resolution=0.3, r_cut=5.0, r_min=1.5, verbose=False)
    g._generate_mesh()
    e = -g._min_dists[g._min_dists > g.r_min]
    e[0] = np.nan
    assert not np.isnan(g.load(e).values).any()


def test_load_checks_length(blocked_grid):
    with pytest.raises(ValueError, match="Expected"):
        blocked_grid.load(np.zeros(3))


def test_configurations_match_energies(blocked_grid):
    cfgs = blocked_grid.prepare_configurations()
    assert len(cfgs) == len(blocked_grid) == blocked_grid.emap.info["grid"]["n_points"]


def test_volumetric_files_are_finite(blocked_grid, tmp_path):
    blocked_grid.emap.write(tmp_path / "x.grd")
    blocked_grid.emap.write(tmp_path / "x.cube")
    lines = (tmp_path / "x.grd").read_text().splitlines()
    assert lines[2].split() == [str(n) for n in blocked_grid.mesh_shape]
    values = np.array(lines[3:], dtype=float)
    assert values.size == np.prod(blocked_grid.mesh_shape)
    assert np.isfinite(values).all() and values.max() < 1e3


def test_radii_without_explicit_generate_mesh():
    g = Grid(nacl(), specie=11, resolution=0.3, r_cut=5.0, r_min=0.5, verbose=False)
    r = g.percolation_radii()
    assert r["r3d"] > 1.5


def test_void_search_finds_octahedral_sites():
    atoms = nacl()
    g = Grid(atoms, specie=11, resolution=0.2, r_cut=5.0, r_min=0.5, verbose=False)
    sites = clustered_local_minima(-g.min_distances, min_distance=2.0,
                                   relative_max_value=0.2, cell=g.cell[:], coord_type="frac")
    assert len(sites) == 4
    na = atoms.get_scaled_positions()[atoms.numbers == 11]
    for frac, _, _ in sites:
        d = (na - frac + 0.5) % 1 - 0.5
        assert np.min(np.linalg.norm(d @ atoms.cell[:], axis=1)) < 0.2
