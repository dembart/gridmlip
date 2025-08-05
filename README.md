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

**gridmlip** is a library for calculating percolation barriers of mobile species in solids using grid-based method with machine learning interatomic potentials (MLIPs).


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

#### Step #1: Construct configurations for processing with your favorite MLIP
```python
from gridmlip import Grid

atomic_types_mapper = {3:0, 31:1, 17:2}
grid = Grid.from_file('your.cif', specie = 3, r_min = 1.8, 
                    atomic_types_mapper=atomic_types_mapper # optional
                  )
cfgs = grid.construct_configurations('data.cfg')
```

#### Step #2: Evaluate the configurations with your favorite MLIP

```
mlp calculate_efs p.mtp data.cfg --output_filename=processed_data.cfg'
```

#### Step #3: Read processed configrations and calculate the percolation barriers

```python
from gridmlip import Grid

grid = Grid.from_file('your.cif', specie = 3, r_min = 1.8)
grid.read_processed_configurations('processed_data.cfg', format = 'cfg')
barriers = grid.percolation_barriers()
```

#### Step #4: Write .grd or .cube file for visualization in VESTA 3.0

```python
g.write_grd('test.grd')
```

## Notebooks


- [Using TensorNet model pre-trained on MatPES](notebooks/TensorNet_MatPES.ipynb)    
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14gabZ-u19K-_I_e7-g67N8-xtPIN36Nw#sandboxMode=true&scrollTo=5TySp6C6UBoZ)


### How to cite
If you use the gridmlip library, please, consider citing this repository.

