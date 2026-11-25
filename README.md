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

## NEB paths of vacancy-assisted hops

The energy landscape of each hop of a percolation network (from [ions](https://github.com/dembart/ions)),
computed with the hopping ion removed and a vacancy on the target site, gives the initial path of its NEB.
All hops share one supercell, so their NEB energies can be compared.

```python
from ions import Percolator

pl = Percolator(atoms, "Na", upper_bound=5.0)
cutoff = pl.cutoff_search(3, bottleneck_radius=0.5)
edges = pl.unique_edges(cutoff, 0.5)

hops = gm.Hops(atoms, "Na", edges, supercell_cutoff=8.0, resolution=0.2, r_min=0.8, tube=2.5)
maps = hops.evaluate(gm.BVSE("Na+"))                       # or an ASE calculator / predictor
maps.barriers                                              # grid estimates, in the order of edges
maps.write_paths("neb_inputs", format="vasp", n_images=12) # hop_000/00/POSCAR, ... or extxyz

# NEB (keep the given images) -> emins, emaxs per hop, in the order of edges
pl.find_percolation_barriers(cutoff, 0.5, emins, emaxs, method="naive")
```

Only points within `tube` of each hop are evaluated (`Grid.evaluate(region=...)`). A single hop:
`gm.Hop(atoms, source, target, offset).evaluate(model).path().images()`; any map: `emap.path(start, end)`.
External calculations: `hops.prepare_configurations("cfgs.xyz")` and `hops.load("processed.xyz")`.

## Interactive view

```python
emap.view()
emap.view(level=0.6, supercell=(1, 2, 2), theme="light")
emap.view().save("map.html")
emap.view().open()
```

Inline in Jupyter or as a standalone page that works offline: VESTA-style atoms, bonds and polyhedra, and the
map as an isosurface starting at the 1D percolation barrier with 1D/2D/3D presets. Options include
`style="soft"`, `hide=["Li"]`, `hide_bonds=["Li"]`, `polyhedra=["Fe", "P"]`, `along="c"`.
`gridmlip.view.structure_view(atoms)` shows a structure (or a CIF, POSCAR, CHGCAR, ... file) without a map.

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
