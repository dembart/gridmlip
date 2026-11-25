"""gridmlip.network: Voronoi hops, symmetry, periodicity and percolation of a hop network."""
import numpy as np
import pytest

import gridmlip as gm
from gridmlip import BVSE, HopMaps, Hops
from gridmlip.network import expand_edges, percolation_barriers, periodicity, unique_edges, voronoi_edges

from structures import lifepo4, licoo2, limn2o4, nacl


# ------------------------------------------------------------------ periodicity

def test_periodicity_counts_translations_per_component():
    assert periodicity(np.zeros((0, 5), int)) == 0
    assert periodicity([[0, 1, 0, 0, 0]]) == 0                            # a finite pair
    assert periodicity([[0, 0, 1, 0, 0]]) == 1
    assert periodicity([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]) == 2
    assert periodicity([[0, 1, 0, 0, 0], [1, 0, 1, 1, 1]]) == 1          # a zigzag along (1, 1, 1)
    # two 1D chains along different axes are not a 2D network
    assert periodicity([[0, 0, 1, 0, 0], [1, 1, 0, 1, 0]]) == 1
    assert periodicity([[0, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 1, 0, 0, 0]]) == 2


# ------------------------------------------------------------------ Voronoi edges and symmetry

@pytest.mark.parametrize("make, specie", [(nacl, "Na"), (lifepo4, "Li"), (licoo2, "Li"), (limn2o4, "Li")])
def test_voronoi_network_is_periodic_symmetric_and_closed(make, specie):
    atoms = make()
    edges = voronoi_edges(atoms, specie)
    assert periodicity(edges) == 3
    keys = set(map(tuple, edges))
    assert all((t, s, -a, -b, -c) in keys for s, t, a, b, c in edges)    # every hop with its reverse
    unique, inverse = unique_edges(atoms, edges)
    assert len(inverse) == len(edges) and np.array_equal(np.unique(inverse), np.arange(len(unique)))
    network, inv = expand_edges(atoms, unique)
    assert set(map(tuple, network)) == keys                               # symmetry closes the network
    assert set(inv) == set(range(len(unique)))


def test_voronoi_lifepo4_hops():
    atoms = lifepo4()
    edges = voronoi_edges(atoms, "Li")
    frac = atoms.get_scaled_positions()
    lengths = np.linalg.norm((frac[edges[:, 1]] + edges[:, 2:] - frac[edges[:, 0]]) @ atoms.cell, axis=1)
    assert np.allclose(sorted(set(np.round(lengths, 2))), [3.0, 4.69, 5.67], atol=0.01)
    assert len(unique_edges(atoms, edges)[0]) == 3


def test_voronoi_filters_must_keep_the_dimensionality():
    atoms = licoo2()                                                      # Li layers: in-plane faces are large
    with pytest.raises(ValueError, match="percolates in 2D"):
        voronoi_edges(atoms, "Li", min_face=0.05)
    assert periodicity(voronoi_edges(atoms, "Li", min_face=0.05, dim=2)) == 2
    edges = voronoi_edges(nacl(), "Na", max_length=3.5, dim=0)            # no Na-Na hop that short
    assert len(edges) == 0


# ------------------------------------------------------------------ percolation

def test_percolation_barriers_of_a_toy_network():
    # chain along a (hops 0.3 eV above sites at 0), joined along b by 0.8 eV hops, and along c by 2.0 eV
    network = [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    emins, emaxs = [0.0, 0.0, 0.0], [0.3, 0.8, 2.0]
    b = percolation_barriers(network, emins, emaxs)
    assert b == pytest.approx({"e1d": 0.3, "e2d": 0.8, "e3d": 2.0})
    # a deep site off the percolating chain does not set the reference
    network2 = network[:1] + [[1, 2, 0, 0, 0]]
    b = percolation_barriers(network2, [0.0, -5.0], [0.3, 0.1])
    assert b["e1d"] == pytest.approx(0.3) and np.isinf(b["e2d"]) and np.isinf(b["e3d"])


# ------------------------------------------------------------------ Hops

def test_hops_from_voronoi_and_percolation():
    hops = Hops.from_voronoi(lifepo4(), "Li", resolution=0.3)
    assert len(hops) == 3 and hops.periodicity == 3 and len(hops.network) == 32
    assert np.bincount(hops.inverse).tolist() == [8, 16, 8]
    maps = hops.evaluate(BVSE("Li+", verbose=False))
    b = maps.percolation_barriers()
    assert 0 < b["e1d"] < 1.0 < b["e2d"] <= b["e3d"] < np.inf
    assert b["e1d"] == pytest.approx(maps.barriers.min())                 # the 1D chain: one hop type


def test_hops_network_percolation_matches_ions(lfp_ions):
    pl, cutoff, edges = lfp_ions
    hops = Hops(lifepo4(), "Li", edges, resolution=0.3)
    maps = hops.evaluate(BVSE("Li+", verbose=False))
    paths = maps.paths(n_images=2)
    emins, emaxs = [min(p.e_start, p.e_end) for p in paths], [p.level for p in paths]
    ref = pl.find_percolation_barriers(cutoff, 0.5, emins, emaxs, method="naive")
    ours = maps.percolation_barriers()
    for k, v in ref.items():
        assert (np.isinf(ours[k]) if v is None else ours[k] == pytest.approx(v))


def test_percolation_needs_the_hops(tmp_path):
    hops = Hops.from_voronoi(lifepo4(), "Li", resolution=0.4)
    maps = hops.evaluate(BVSE("Li+", verbose=False))
    maps.save(tmp_path)
    assert HopMaps.load(tmp_path, hops).percolation_barriers() == pytest.approx(maps.percolation_barriers())
    with pytest.raises(ValueError, match="needs the Hops"):
        HopMaps(list(maps)).percolation_barriers()


@pytest.fixture(scope="module")
def lfp_ions():
    from ions import Percolator
    pl = Percolator(lifepo4(), "Li", upper_bound=6.3)
    cutoff = pl.cutoff_search(3, 0.5)
    return pl, cutoff, pl.unique_edges(cutoff, 0.5)
