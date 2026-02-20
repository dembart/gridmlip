from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

from ase.io import read, write, cube

from .utils import read_cfg, write_cfg
from ._neighborhood import nn_list
from ._percolation_analysis import Percolyze
from ._mesh import Mesh



__author__ = "Artem Dembitskiy"


@dataclass
class Grid:
    """
    Grid object representing a mesh for atomic structures.

    Parameters
    ----------
    atoms : ASE Atoms object
        Atomic structure.
        
    specie : int
        Atomic number of the mobile specie.
        
    resolution : float, default 0.25
        Spacing between points (in Angstroms).
        
    r_cut : float, default 5.0
        Cutoff radius to find the first nearest neighbor for each grid point.
        
    r_min : float, default 1.5
        Blocking sphere radius.
        
    atomic_types_mapper : dict, optional
        Mapper of atomic numbers into species used by MLIP.
        
    symprec : float or None, default 0.1
        Symmetry precision for irreducible mesh points.
        
    empty_framework : bool, default True
        Remove mobile types of interest from the structure.
        
    verbose : bool, default True
        Print detailed output during mesh generation.
        
    """
    
    atoms: object
    specie: int
    resolution: float = 0.25
    r_cut: float = 5.0
    r_min: float = 1.5
    symprec: float = 0.1
    atomic_types_mapper: dict = None
    empty_framework: bool = True
    verbose: bool = True

    cell: object = field(init=False)
    base: object = field(init=False)
    _mesh: object = field(default=None, init=False)

    
    def __post_init__(self):
        
        if self.atomic_types_mapper:
            numbers = self._map_atomic_types(self.atomic_types_mapper, self.atoms.numbers)
            self.atoms = self.atoms.copy()
            self.atoms.numbers = numbers
            self.specie = self.atomic_types_mapper[self.specie]
        self.atoms = self.atoms.copy()
        self.cell = self.atoms.cell
        self.base = self.atoms[self.atoms.numbers != self.specie] if self.empty_framework else self.atoms.copy()

    def __repr__(self):
        return (f"Grid(atoms='{self.atoms.symbols}', specie={self.specie}, "
                f"resolution={self.resolution}, r_cut={self.r_cut}, "
                f"r_min={self.r_min}, symprec={self.symprec}, "
                f"empty_framework={self.empty_framework})")
    
    def generate_mesh(self):      
        
        self._mesh = Mesh.from_atoms(self.base,
                               resolution = self.resolution,
                               symprec = self.symprec)
        self.mesh_shape = self._mesh.mesh_shape
        self.ir_grid_points, self.mapping = self._mesh.get_irreducible_mapping()
        self.mesh_frac = self._mesh.addresses / (np.array(self._mesh.mesh_shape) - 1)
        self.mesh_cart = np.dot(self.mesh_frac, self.base.cell)
        
        self.min_dists, _ = nn_list(self.base.positions,
                                    self.mesh_cart[self.ir_grid_points],
                                    self.r_cut,
                                    self.cell)
        if self.verbose:
            total_points = self.mesh_frac.shape[0]
            irreducible_points = self.ir_grid_points.shape[0]
            passed_filter = (self.min_dists > self.r_min).sum()
            compression = total_points / passed_filter if passed_filter else np.nan
        
            print("Mesh Summary")
            print("────────────────────────────────────")
            print(f"  Mesh shape:                         {tuple(self.mesh_shape)}")
            print(f"  Symops:                             {len(self._mesh.rotations)}")
            print(f"  Total points:                       {total_points:,}")
            print(f"  Irreducible points:                 {irreducible_points:,}")
            print(f"  Irreducible points (r_min filter):  {passed_filter:,}")
            print(f"  Compression ratio:                  {compression:6.2f}×")
            print("────────────────────────────────────\n")

        

    def _map_atomic_types(self, atomic_types_mapper, numbers):
        u,inv = np.unique(numbers,return_inverse = True)
        return np.array([atomic_types_mapper[x] for x in u])[inv].reshape(numbers.shape)

    
    @classmethod
    def from_file(cls, file,
                  specie,
                  resolution = 0.25,
                  r_cut = 5.0,
                  r_min = 1.5, 
                  symprec = 0.1,
                  atomic_types_mapper = None, 
                  empty_framework = True,
                  verbose=False
                 ):
        """ 
        Create Grid object from the file.

        Parameters
        ----------
        
        file: string
            .xyz of .cfg file representing atomic structure

        specie: int
            atomic number of the mobile specie

        resolution: float, 0.2 by default
            spacing between points (in Angstroms)

        r_cut: float
            cutoff radius to find the first nearest neighbor for each grid point

        r_min: float
            blocking sphere radius

        atomic_types_mapper: dict, optional
            mapper of the species into atomic numbers

        empty_framework: boolean, True by default
            whether to remove mobile types of interest from the structure

        symprec: None or float, 0.1 by default
            symmetry precision
            used to get symmetry operations for finding the irreducible mesh points
            may reduce number of MLIP calculations by factor of 10-100
            if None, the symmetry will not be used.

        Returns
        -------
        Grid object
        """

        if file.split('.')[-1] == 'cfg':
            atoms = read_cfg(file)[0]
        else:
            atoms = read(file)
        return cls(atoms, specie, resolution=resolution,
                   symprec=symprec, r_cut=r_cut, r_min=r_min,
                   atomic_types_mapper=atomic_types_mapper,
                   empty_framework=empty_framework, verbose=verbose
                  )


    
    def construct_configurations(self, config_format = 'ase', filename = None):
        """ 
        Construct atomic configurations for further calculations.

        Parameters
        ----------
        
        filename: string (Optional)
            path to save .xyz of .cfg file with atomic configurations
        
        config_format: string, "ase" by default, can be "pymatgen" or "ase"
            data format of the created configurations (pymatgen's Structure of ASE's Atoms)

        
        Returns
        -------
        configurations: list of ASE's atoms object
        """

        if self._mesh is None:
            self.generate_mesh()
        
        configurations = []
        if config_format == 'ase':

            ir_mesh_cart = self.mesh_cart[self.ir_grid_points]
            assert len(ir_mesh_cart) == len(self.min_dists)
            for p, d in tqdm(zip(ir_mesh_cart, self.min_dists), desc = 'creating configurations'):
                if d > self.r_min:
                    framework = self.base.copy()
                    framework.append(self.specie)
                    framework.positions[-1] = p
                    configurations.append(framework)
                    
        elif config_format == 'pymatgen':

            from pymatgen.io.ase import AseAtomsAdaptor
            base = AseAtomsAdaptor.get_structure(self.base)
            ir_mesh_frac = self.mesh_frac[self.ir_grid_points]
            for p, d in tqdm(zip(ir_mesh_frac, self.min_dists), desc = 'creating configurations'):
                if d > self.r_min:
                    framework = base.copy()
                    framework.append(self.specie, coords = p)
                    configurations.append(framework)
        else:
            
            raise ValueError(f"Wrong config_format {config_format}")
        
        if filename:
            if config_format != 'ase':
                raise ValueError('Only "ase" format is allowed for saving files')
            if filename.split('.')[-1] == 'cfg':
                write_cfg(filename, configurations)
            else:
                write(filename, configurations)
        return configurations



    def load_energies(self, energies):

        """ 
        Load energies obtained by any MLIP or structure-to-property model.
        Should match order in created configurations.

        Parameters
        ----------
        
        energies: np.array
            calculated energies for the created configurations
        """

        if self._mesh is None:
            self.generate_mesh()

        self.energies = energies
        
        ir_distribution = np.ones_like(self.min_dists) * np.inf
        ir_distribution[self.min_dists > self.r_min] = self.energies
        
        self.distribution = np.nan_to_num(ir_distribution, copy = False, nan = np.inf)[self.mapping]
        self.data = self.distribution.reshape(self.mesh_shape)


    def read_processed_configurations(self, filename, file_format = 'xyz'):

        """ 
        Read processed (by any MLIP) atomic configurations
        with calculated energies.

        Parameters
        ----------
        
        filename: string
            path to the processed .xyz of .cfg file

        file_format: string, 'xyz' by default
            format of the file
            if 'cfg' will use custom function read_cfg() to read MLIP 2.0 generated data
            otherwise will use ASE's read()
        """

        if file_format == 'cfg':
            atoms_list = read_cfg(filename)
        else:
            atoms_list = read(filename, index = ':')
        energies = np.array([atoms.get_potential_energy() for atoms in atoms_list])
        del atoms_list
        self.load_energies(energies)



    def write_cube(self, filename):

        """
        Write .cube file containing structural and MLIP distribution data.

        Parameters
        ----------

        filename: str
            file name to write .cube
        """

        data = self.data
        nx, ny, nz = data.shape
        with open(f'{filename}.cube', 'w') as f:
            cube.write_cube(f, self.atoms, data = data[:nx-1, :ny-1, :nz-1])
    


    def write_grd(self, filename):
        
        """
        Write MLIP distribution volumetric file for VESTA 3.0.

        Parameters
        ----------

        filename: str
            file name to write .grd
        """

        
        data = self.data
        voxels = data.shape[0] - 1, data.shape[1] - 1, data.shape[2] - 1
        cellpars = self.cell.cellpar()
        with open(f'{filename}.grd' , 'w') as report:
            comment = '# MLIP data made with gridmlip package: https://github.com/dembart/gridmlip'
            report.write(comment + '\n')
            report.write(''.join(str(p) + ' ' for p in cellpars).strip() + '\n')
            report.write(''.join(str(v) + ' ' for v in voxels).strip() + '\n')
            for i in range(voxels[0]):
                for j in range(voxels[1]):
                    for k in range(voxels[2]):
                        val = data[i, j, k]
                        report.write(str(val) + '\n')



    def percolation_barriers(self, encut=10.0, n_jobs=1, backend='threading'):
        """
        Find percolation energy and dimensionality of a migration network.

        Parameters
        ----------

        encut: float, 5.0 by default
            cutoff energy above which barriers supposed to be np.inf

        n_jobs: int, 1 by default
            number of jobs to run for percolation energy search

        backend: str, 'threading' by default
            see joblib's documentations for more details

        Returns
        ----------
        
        energies: dict
            infromation about percolation {'e1d': float, 'e2d': float, 'e3d': float}
        """
        pl = Percolyze(self.data)
        return pl.percolation_barriers(encut = encut, n_jobs=n_jobs, backend=backend)


        
    def __repr__(self):
        return (
            f'Grid(atoms={self.atoms.symbols}, specie={self.specie}, '
            f'resolution={self.resolution}, r_cut={self.r_cut}, '
            f'r_min={self.r_min}, symprec={self.symprec})'
        )