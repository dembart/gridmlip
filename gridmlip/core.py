import numpy as np
from tqdm import tqdm
from ase.io import read, write, cube
from pymatgen.io.ase import AseAtomsAdaptor
from .utils import read_cfg, write_cfg
from ._neighborhood import nn_list
from ._percolation_analysis import Percolyze



__version__ = "0.1.1"



class Grid:

    def __init__(self, atoms, specie, resolution = 0.25,
                 r_cut = 5.0, r_min = 1.5, atomic_types_mapper = None, empty_framework = True):

        """ 
        Initialization. 

        Parameters
        ----------
        
        atoms: ASE's Atoms object
            atomic structure

        specie: int
            atomic number of the mobile specie

        resolution: float, 0.2 by default
            spacing between points (in Angstroms)

        r_cut: float
            cutoff radius to find the first nearest neighbor for each grid point

        r_min: float
            blocking sphere radius

        atomic_types_mapper: dict, optional
            mapper of atomic numbers into species used by MLIP

        empty_framework: boolean, True by default
            whether to remove mobile types of interest from the structure
        """

        if atomic_types_mapper:
            numbers = self._map_atomic_types(atomic_types_mapper, atoms.numbers)
            atoms = atoms.copy()
            atoms.numbers = numbers
            specie = atomic_types_mapper[specie]
        self.atoms = atoms.copy()
        self.specie = specie
        self.resolution = resolution
        self.cell = self.atoms.cell
        self._mesh(resolution)
        self.base = self.atoms[self.atoms.numbers != self.specie] if empty_framework else atoms.copy()
        self.min_dists, _ = nn_list(self.base.positions, self.mesh_cart, r_cut, self.cell)
        self.r_cut = r_cut
        self.r_min = r_min



    def _map_atomic_types(self, atomic_types_mapper, numbers):
        u,inv = np.unique(numbers,return_inverse = True)
        return np.array([atomic_types_mapper[x] for x in u])[inv].reshape(numbers.shape)



    def _mesh(self, resolution):
        
        """ 
        This method creates grid of equidistant points in 3D
        with respect to the input resolution. 

        Parameters
        ----------
        resolution: float, 0.2 by default
            spacing between points (in Angstroms)

        Returns
        -------
        mesh_cart: np.array
            cartesian coordinates of meshgrid
        
        """
        
        a, b, c, _, _, _ = self.cell.cellpar()
        nx, ny, nz = int(a // resolution), int(b // resolution), int(c // resolution)
        x = np.linspace(0, 1, nx) 
        y = np.linspace(0, 1, ny) 
        z = np.linspace(0, 1, nz)
        self.mesh_frac = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        self.mesh_cart = self.cell.cartesian_positions(self.mesh_frac)
        self.size = (nx, ny, nz)
        return self.mesh_cart



    @classmethod
    def from_file(cls, file, specie, resolution = 0.25,
                  r_cut = 5.0, r_min = 1.5, atomic_types_mapper = None, empty_framework = True):
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

        Returns
        -------
        Grid object
        """

        if file.split('.')[-1] == 'cfg':
            atoms = read_cfg(file)[0]
        else:
            atoms = read(file)
        return cls(atoms, specie, resolution=resolution,
                   r_cut = r_cut, r_min = r_min, atomic_types_mapper =  atomic_types_mapper, empty_framework=empty_framework)



    def construct_configurations(self, format = 'ase', filename = None):
        """ 
        Construct atomic configurations for further calculations.

        Parameters
        ----------
        
        filename: string (Optional)
            path to save .xyz of .cfg file with atomic configurations
        
        format: string, "ase" by default, can be "pymatgen" or "ase"
            data format of the created configurations (pymatgen's Structure of ASE's Atoms)

        
        Returns
        -------
        configurations: list of ASE's atoms object
        """
        
        configurations = []
        if format == 'ase':
            for p, d in tqdm(zip(self.mesh_cart, self.min_dists), desc = 'creating configurations'):
                if d > self.r_min:
                    framework = self.base.copy()
                    framework.append(self.specie)
                    framework.positions[-1] = p
                    configurations.append(framework)
        elif format == 'pymatgen':
            base = AseAtomsAdaptor.get_structure(self.base)
            for p, d in tqdm(zip(self.mesh_frac, self.min_dists), desc = 'creating configurations'):
                if d > self.r_min:
                    framework = base.copy()
                    framework.append(self.specie, coords = p)
                    configurations.append(framework)
        else:
            raise ValueError(f"Wrong format {format}")
        
        if filename:
            if format != 'ase':
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

        self.energies = energies
        self.distribution = np.ones_like(self.min_dists) * np.inf
        self.distribution[self.min_dists > self.r_min] = self.energies
        self.distribution = np.nan_to_num(self.distribution, copy = False, nan = np.inf)
        self.data = self.distribution.reshape(self.size)


    def read_processed_configurations(self, filename, format = 'xyz'):

        """ 
        Read processed (by any MLIP) atomic configurations
        with calculated energies.

        Parameters
        ----------
        
        filename: string
            path to the processed .xyz of .cfg file
        """

        if format == 'cfg':
            atoms_list = read_cfg(filename)
        else:
            atoms_list = read(filename, index = ':')
        self.energies = np.array([atoms.get_potential_energy() for atoms in atoms_list])
        del atoms_list
        self.distribution = np.ones_like(self.min_dists) * np.inf
        self.distribution[self.min_dists > self.r_min] = self.energies
        self.distribution = np.nan_to_num(self.distribution, copy = False, nan = np.inf)
        self.data = self.distribution.reshape(self.size)



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



    def percolation_barriers(self, encut = 10.0):
        """
        Calculate percolation barriers.

        Parameters
        ----------
        encut: float, 10.0 by default
            upper bound for searching the barrier
        """
        pl = Percolyze(self.data)
        return pl.percolation_barriers(encut = encut)