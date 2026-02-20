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
Here we describe the pipeline in general. For a specific example, see [Notebooks](#notebooks).

```python 
from gridmlip import Grid
from gridmlip.integrations.sevennet import evaluate_atoms_list # sevenn must be installed

file = './data/Li10Ge(PS6)2_mp.cif'
specie = 3

### Create grid
g = Grid.from_file(
    file,
    specie,
    r_min=0.8,
    resolution=0.2,
    symprec=0.1,
    empty_framework=True,
    verbose=True
)

### Prepare inequivalent atomic configurations
atoms_list = g.construct_configurations(config_format='ase')

### Predict energies
energy_list, forces_list = evaluate_atoms_list(
    atoms_list,
    model_path='.data/checkpoint_sevennet_0.pth',
    device='cuda',
    compute_force=False,
    batch_size=256
)

### Load energies
g.load_energies(energy_list)

### Determine percolation barriers
barriers = g.percolation_barriers(n_jobs=-1)

### Save .grd file for visualization in VESTA 3.0
g.write_grd('test.grd')
```

## Notebooks


- [Using SevenNet for calculating percolation barriers](notebooks/integrations.ipynb)    

- [Using moment tensor potentials (as implemented in MLIP 2.0) for calculating percolation barriers](notebooks/mtp.ipynb)


### How to cite
If you use the gridmlip library, please, consider citing this repository.

