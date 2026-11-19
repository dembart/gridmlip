## gridmlip
<p align="left">
<a href="https://github.com/dembart/gridmlip/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-darkred"></a>


## Contents
- [About](#about)
- [Installation](#installation)
- [How to use](#how-to-use)
- [Notebooks](#notebooks)
- [How to cite](#how-to-cite)


## About

**gridmlip** is a library for the symmetry-aware grid-based sampling of the energy landscape of mobile species in solids using machine learning interatomic potentials (MLIPs).

> (!) Note: The API and functionality may change. Errors are expected. The library is under development.

## Installation

```
pip install gridmlip
```
or

```python
git clone https://github.com/dembart/gridmlip
cd gridmlip
pip install .
```

## How to use

Two objects: a `Grid` says **where** to put the mobile ion; evaluating a model on it gives an `EnergyMap`.

### Percolation barriers with a universal MLIP

```python
from ase.io import read
from mace.calculators import mace_mp
import gridmlip as gm

atoms = read("Li10Ge(PS6)2_mp.cif")
grid = gm.Grid(atoms, specie="Li", resolution=0.2, r_min=0.8)

emap = grid.evaluate(mace_mp(model="medium"))   # MACE / SevenNet calculators are batched automatically
emap.percolation_barriers()                     # {'e1d': ..., 'e2d': ..., 'e3d': ...}
emap.write("li.grd")                            # VESTA volumetric file (.grd or .cube)
emap.save("li.npz")                             # reload with gm.EnergyMap.load("li.npz")
```

Batched predictors with explicit settings:

```python
from gridmlip.predictors import mace, sevennet

emap = grid.evaluate(mace("medium-mpa-0", device="cuda"))          # name or path; float32 by default
emap = grid.evaluate(mace("my_finetune.model", head="pbe"))        # multi-head models need head=
emap = grid.evaluate(sevennet("7net-mf-ompa", modal="mpa"))
emap = grid.evaluate(my_function)                                  # any callable list[Atoms] -> energies
```

### Calculations run elsewhere (MTP, DFT, cluster jobs)

```python
cfgs = grid.prepare_configurations(supercell_cutoff=5.0, filename="cfgs.cfg")
# ... compute the energies ...
emap = grid.load_energies("processed.cfg")      # or an array of energies in the same order
emap.percolation_barriers()
```

### Bond valence site energy

```python
emap = gm.bvse(atoms, "Na+", resolution=0.2)
emap.percolation_barriers()
emap.info["oxidation_states"]                   # charges used
```

Oxidation states are taken from `atoms.arrays["oxi_states"]` if present, otherwise assigned with
pymatgen's BVAnalyzer (as in bvlain) and printed. The same model on an explicit grid:

```python
emap = gm.Grid(atoms, "Na", resolution=0.2, empty_framework=False).evaluate(gm.BVSE("Na+"))
```

### Local minima and void space

```python
emap.local_minima(min_distance=2.0, energy_window=0.5)   # [(position, energy, n_points), ...]
grid.percolation_radii()                                 # largest percolating free sphere
grid.min_distances                                       # distance to the host, 3D array
```

### Defect configurations for MLIP training

```python
from gridmlip.defects import interstitials, vacancies, frenkel_pairs

params = dict(resolution=0.25, r_min=1.3, r_cut=7.5)
cfgs = interstitials(atoms, 3, **params)
cfgs += vacancies(atoms, 3, upper_bound=6.0, **params)
cfgs += frenkel_pairs(atoms, 3, **params)
```

Each configuration has `atoms.info["defect"]` set to `"interstitial"`, `"vacancy"` or `"frenkel_pair"`.

### Training sets: defects, selection, perturbation

From the shell (one `.xyz` per material, plus `config.yaml` and `summary.csv`):

```bash
gridmlip dataset --print-config > dataset.yaml      # all settings with defaults; edit
gridmlip dataset structures.xyz --config dataset.yaml --specie 3 --output dataset/
gridmlip dataset structures.xyz --config dataset.yaml --dry-run      # candidate counts only
```

Reruns skip finished materials. On a cluster, split the work with a job array; the result does not depend on the split:

```bash
#SBATCH --array=0-31
gridmlip dataset structures.xyz --config dataset.yaml --output dataset/ --chunk ${SLURM_ARRAY_TASK_ID}/32
```

The same recipe in Python, built from the public functions (identical files for the same settings):

```python
from pathlib import Path
from ase.io import write
from gridmlip.defects import interstitials, vacancies, frenkel_pairs
from gridmlip.dataset import soap_features, select_fps, rattle, save_settings

seed, out = 42, Path("dataset"); out.mkdir(exist_ok=True)
p = dict(resolution=0.25, r_min=1.5, r_cut=5.0, symprec=0.1)
save_settings(out / "config.yaml", seed=seed, specie=3, defects=p)   # + library versions

for atoms in atoms_list:                                  # atoms.info["material_id"] names the files
    mid = atoms.info["material_id"]
    cfgs = (interstitials(atoms, 3, **p)
            + vacancies(atoms, 3, max_sep_dist="auto", upper_bound=6.0, **p)
            + frenkel_pairs(atoms, 3, max_sep_dist="auto", upper_bound=6.0, **p))
    X = soap_features(cfgs, center=-1, r_cut=6.0, n_max=3, l_max=2, sigma=0.3)
    idx = select_fps(X, pca=0.95, max_samples=500, rel_tol=1e-5)
    train = [rattle(cfgs[j], seed=seed, pos_std=0.05, strain_std=0.02, d_min=1.0) for j in idx]
    write(out / f"dataset_{mid}.xyz", train)
```

`rattle` is the only random step. Its random stream is derived from the seed, `material_id` and the
configuration itself, so the same structure always gets the same perturbation, regardless of the order
of structures, the split into chunks, or the generation settings. Every output records
`rattle_seed`, `rattle_key` and `rattle_copy`; `rattle(cfg, seed=42, key=j)` reproduces datasets made with 0.2.

### Migrating from 0.2

| 0.2 | 0.3 |
|---|---|
| `grid.construct_cfgs()` | `grid.prepare_configurations()` |
| `grid.load_energies(e); grid.percolation_barriers()` | `grid.load_energies(e).percolation_barriers()` |
| `grid.read_processed_cfgs(f)` | `grid.load_energies(f)` |
| `grid.write_grd("x")` | `emap.write("x.grd")` |
| `bvse_distribution(atoms, "Li+")` | `gm.bvse(atoms, "Li+")` |
| `integrations.mace.evaluate_atoms_list(...)` | `grid.evaluate(predictors.mace(...))` |
| `DefectGenerator(...).generate_dataset("int")` | `defects.interstitials(...)` |
| `dataset.generate_defects_dataset(...)` | `gridmlip dataset ...` or the recipe above |

The 0.2 names still work in 0.3 and emit a `DeprecationWarning`; they will be removed in 0.4.


## Notebooks


- [Using SevenNet for calculating percolation barriers](notebooks/integrations.ipynb)    
- [Using moment tensor potentials (as implemented in MLIP 2.0) for calculating percolation barriers](notebooks/mtp.ipynb)


### How to cite
If you use the gridmlip library, please, consider citing this repository.

