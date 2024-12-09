## gridmlip
<p align="left">
<a href="https://github.com/dembart/gridmlip/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-darkred"></a>


## Contents
- [About](#about)
- [Installation](#installation)
- [How to use](#how-to-use)
- [Citation](#how-to-cite)


## About

gridmlip is a library for calculating 1-3D percolation barriers of mobile species in solids using grid-based machine learning interatomic potential (MLIP) method.

### Installation

```python
git clone https://github.com/dembart/gridmlip
cd gridmlip
pip install .
```

## How to use

#### Step #1: Construct configurations for processing with your favorite MLIP
```python
from gridmlip import Grid

atomic_types_mapper = {3:0, 31:1, 17:2}
g = Grid.from_file('1.cif', specie = 3, r_min = 1.8, atomic_types_mapper=atomic_types_mapper)
_ = g.construct_configurations('data.cfg')
del _

```

#### Step #2: Evaluate the configurations with your favorite MLIP

```
mlp calculate_efs p.mtp data.cfg --output_filename=processed_data.cfg'
```

#### Step #3: Read processed configrations and calculate the percolation barriers

```python
from gridmlip import Grid

g = Grid.from_file('1.cif', specie = 3, r_min = 1.8)
g.read_processed_configurations('processed_data.cfg_0', format = 'cfg')
g.percolation_barriers()
```

#### Step #4: Write .grd or .cube file for visualization in VESTA 3.0

```python
g.write_grd('test.grd')
```


### How to cite
If you use the gridmlip library, please, consider citing our paper 
```
@article{none,
      title={none}, 
      author={none},
      year={2025},
      eprint={none},
      archivePrefix={none},
      primaryClass={none},
      url={none}, 
}
```

