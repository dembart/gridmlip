## gridmlip
<p align="left">
<a href="https://github.com/dembart/gridmlip/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-darkred"></a>

**gridmlip** samples the energy landscape of a mobile ion in a crystal on a symmetry-reduced grid,
using machine learning interatomic potentials (MLIPs) or bond valence site energies, and finds
percolation barriers. It also generates defect configurations and training sets for MLIPs.

> The library is under development; the API may change.

## Contents
- [Installation](#installation)
- [Examples](#examples)
    - [Percolation barriers with a batched MLIP](#percolation-barriers-with-a-batched-mlip)
    - [ASE calculators](#ase-calculators)
    - [External calculations (MTP, DFT, cluster jobs)](#external-calculations-mtp-dft-cluster-jobs)
    - [Bond valence site energy](#bond-valence-site-energy)
    - [Forces, stress and descriptors](#forces-stress-and-descriptors)
    - [Saving and visualization](#saving-and-visualization)
    - [Local minima and percolation radii](#local-minima-and-percolation-radii)
    - [Evaluating part of the cell](#evaluating-part-of-the-cell)
    - [Paths between two points](#paths-between-two-points)
    - [NEB paths of vacancy-assisted hops](#neb-paths-of-vacancy-assisted-hops)
    - [Interactive view](#interactive-view)
    - [Defect configurations](#defect-configurations)
    - [Training datasets](#training-datasets)
- [Notebooks](#notebooks)
- [How to cite](#how-to-cite)

## Installation

```bash
pip install gridmlip
pip install "gridmlip[fast]"     # numba: 10-100x faster percolation analysis
pip install "gridmlip[ml]"       # training-set tools: SOAP, FPS, rattling
```

MLIPs are used if installed: `pip install mace-torch`, `sevenn` or `gnn_lib`; any ASE calculator works.

## Percolation barriers with a batched MLIP

```python
from ase.io import read
import gridmlip as gm
from gridmlip.predictors import mace

atoms = read("Li10Ge(PS6)2_mp.cif")
grid = gm.Grid(atoms, specie="Li", resolution=0.2, r_min=0.8)

emap = grid.evaluate(mace("medium-mpa-0", device="cuda"))
emap.percolation_barriers()
```

Other batched predictors:

```python
from gridmlip.predictors import sevennet, gnn_lib

emap = grid.evaluate(sevennet("7net-mf-ompa", modal="mpa"))
emap = grid.evaluate(gnn_lib("cgcnn.pth", r_cut=5.0))
emap = grid.evaluate(my_function)  # any callable: list[Atoms] -> energies
```

## ASE calculators

MACE and SevenNet calculators are batched automatically; other calculators run one structure at a time.

```python
from mace.calculators import mace_mp

emap = grid.evaluate(mace_mp(model="medium"))
```

## External calculations (MTP, DFT, cluster jobs)

```python
grid.prepare_configurations(supercell_cutoff=5.0, filename="cfgs.cfg")
```

```python
emap = grid.load("processed.cfg")
emap.percolation_barriers()
```

## Bond valence site energy

```python
emap = gm.bvse(atoms, "Li+", resolution=0.2)
emap.percolation_barriers()
```

Oxidation states are taken from `atoms.arrays["oxi_states"]` or assigned with pymatgen's BVAnalyzer.

## Forces, stress and descriptors

```python
pred = mace("medium-mpa-0", device="cuda")
out = pred.predict(grid.prepare_configurations(), ("energy", "forces", "stress", "descriptors"))
```

## Saving and visualization

```python
emap.write("li.grd")
emap.save("li.npz")
emap = gm.EnergyMap.load("li.npz")
```

`.grd` and `.cube` files open in VESTA.

## Local minima and percolation radii

```python
emap.local_minima(min_distance=2.0, energy_window=0.5)
grid.percolation_radii()
```

## Evaluating part of the cell

Only points inside `region` are evaluated: a boolean array of shape `grid.mesh_shape`, or a
function of the Cartesian positions of the mesh points. Their energies also fill the symmetry-equivalent
points; all other points are `+inf`.

```python
center = atoms.positions[0]
region = lambda pts: np.linalg.norm(pts - center, axis=1) < 3.0
emap = grid.evaluate(calc, region=region)

grid.region_points(region)                          # the points that will be evaluated
grid.prepare_configurations(region=region)          # external calculations of a region...
emap = grid.load("processed.xyz", region=region)    # ...and their energies
```

## Paths between two points

Lowest-barrier path of the mobile ion between two sites of a map: the lowest energy at which
they are connected, and the shortest route through points at or below it, resampled to
equally spaced points.

```python
(p0, e0, _), (p1, e1, _) = emap.local_minima(min_distance=2.0)[:2]
path = emap.path(p0, p1, n_images=12)
path.barrier, path.level, path.saddle, path.points   # eV, eV, (3,), (12, 3)
```

`p1 - p0` must be the displacement itself (shorter than half the cell).

## NEB paths of vacancy-assisted hops

Initial NEB paths for the hops of a percolation network. For each hop the mobile ion is sampled in the
hop's supercell without the hopping ion and with a vacancy on the target site (the other mobile ions stay),
only in a tube around the hop; its lowest-barrier path gives the NEB images. All hops share one supercell,
so their NEB energies can be compared, and every result keeps the order of the edges.

```python
hops = gm.Hops.from_mincut_maxdim(atoms, "Na", bottleneck_radius=0.5)   # shortest cutoff of the highest dimensionality
hops = gm.Hops.from_voronoi(atoms, "Na", min_face=0.02, max_length=8.0) # or Voronoi neighbours of the mobile sublattice
hops = gm.Hops(atoms, "Na", edges)                                      # or your own [source, target, ox, oy, oz]

hops.summary()
maps = hops.evaluate(gm.BVSE("Na+"))                       # or an ASE calculator / predictor, as Grid.evaluate
maps.barriers                                              # grid estimates (eV)
maps.percolation_barriers()                                # {'e1d', 'e2d', 'e3d'} of the hop network (eV)
paths = maps.paths(n_images=12)                            # paths[i].images(): list of Atoms for edges[i]
maps.write_paths("neb_inputs", format="vasp")              # hop_000/00/POSCAR, ...; or "extxyz" + paths.json
maps[0].view(path=maps[0].path(n_images=9))                # one hop, centered, its NEB images as a trajectory

# NEB energies instead of the grid estimates (one per hop of `hops`)
gm.network.percolation_barriers(hops.network, emins[hops.inverse], emaxs[hops.inverse])
```

Every option of `Hops` (`supercell_cutoff`, `resolution`, `r_min`, `tube`) also goes to `from_mincut_maxdim`
and `from_voronoi`. `from_voronoi` ignores the framework; `min_face` drops hops through faces smaller than that
fraction of the site's Voronoi surface (taken back if the network would lose a dimension), `max_length` drops
longer hops. The barrier is the lowest level at which the hops percolate minus the
lowest site energy of the percolating component.

One hop:

```python
hop = gm.Hop(atoms, source=0, target=7, offset=(0, 0, -1), supercell_cutoff=8.0)
hmap = hop.evaluate(gm.BVSE("Na+"), resolution=0.2, tube=2.5)
hmap.write("hop.grd")
path = hmap.path(n_images=12)
path.write("path.xyz")
```

- Keep the images as given in the NEB (no re-interpolation); IDPP is not needed.
- `tube` (Angstrom) bounds the evaluated region; raise it if `path()` reports that the sites are not connected.
- For MLIPs use a larger `r_min` (~1.2) to skip very close contacts, and mind the cost: every point
  is one evaluation of the host supercell; a coarser `resolution` or a narrower `tube` cuts it.
- External calculations: `hops.prepare_configurations("cfgs.xyz")` (`info["hop_index"]`, `info["grid_index"]`),
  then `hops.load("processed.xyz")`. Maps: `maps.save("maps")`, `gm.HopMaps.load("maps", hops)`.

## Interactive view

```python
emap.view()                                                # inline in Jupyter
emap.view(level=0.6, supercell=(1, 2, 2), theme="light", hide_bonds=["Li"], polyhedra=["Fe", "P"])
emap.view().save("map.html")                               # standalone page, works offline
emap.view().open()

from gridmlip.view import structure_view
structure_view(atoms)                                      # a structure or a CIF, POSCAR, ... file
```

## Defect configurations

```python
from gridmlip.defects import interstitials, vacancies, frenkel_pairs

params = dict(resolution=0.25, r_min=1.3, r_cut=7.5)
cfgs = interstitials(atoms, 3, **params) + vacancies(atoms, 3, **params) + frenkel_pairs(atoms, 3, **params)
```

## Training datasets

Defects → SOAP → farthest point sampling → reproducible rattling, one `.xyz` per material.

```bash
gridmlip dataset --print-config > dataset.yaml
gridmlip dataset structures.xyz --config dataset.yaml --output datasets/li --dry-run
gridmlip dataset structures.xyz --config dataset.yaml --output datasets/li
```

```python
from gridmlip.dataset import build_dataset

build_dataset("structures.xyz", config="dataset.yaml", output_dir="datasets/li")
```

On a cluster:

```bash
#SBATCH --array=0-31
gridmlip dataset structures.xyz --config dataset.yaml --output datasets/li --chunk ${SLURM_ARRAY_TASK_ID}/32
```

## Notebooks

- [Percolation barriers with a batched MLIP](notebooks/01_mlip_barriers.ipynb)
- [Bond valence site energy](notebooks/02_bvse.ipynb)
- [Training dataset for an MLIP](notebooks/03_dataset.ipynb)
- [Interactive view of energy maps](notebooks/04_view.ipynb)

## How to cite

If you use the gridmlip library, please consider citing this repository.
