import itertools
import numpy as np
from scipy.ndimage import measurements, generate_binary_structure
from joblib import Parallel, delayed


class Percolyze:

    def __init__(self, data):
        self.data = data
        self.size = data.shape
    
    def _cross_boundary(self, coords, data_shape):
        
        """ 
        Check if connected component crosses the boundary of unit cell

        Parameters
        ----------

        coords: np.array
            coordinates of points in connected component
            
        data_shape: list
            shape of the mesh constructed over supercell
        
        Returns
        ----------
        
        d: int
            number of unit cells within a supercell that contains connected component
        """

        probe = coords[0, :]
        cell_location = np.floor(probe / data_shape)
        translations = np.array(list(itertools.product([0, 1],
                                                       [0, 1],
                                                       [0, 1])))
        translations = translations - cell_location
        test = probe + translations * data_shape
        d = np.argwhere(abs(coords[:, None] - test).sum(axis = 2) == 0).shape[0]
        return d



    def _connected_components(self, data, tr): 

        """ 
        Find connected components

        Parameters
        ----------

        data: np.array
            BVSE distribution data
            
        tr: float
            energy threshold to find components

        task: str, either "bvse" or "void"
            select type of calculation 
         
        Returns
        ----------
        
        labels, features: np.array, number of components
            labels are data points colored to features values
        """

        n = 2
        lx, ly, lz = data.shape
        superdata = np.zeros((n * lx, n * ly, n * lz))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    superdata[i*lx:(i+1)*lx, j*ly:(j+1)*ly, k*lz:(k+1)*lz] = data

        region = superdata - superdata.min()
        structure = generate_binary_structure(3,3)
        
        labels, features = measurements.label(region < tr, structure = structure)
        
        labels_with_pbc = self._apply_pbc(labels)
        return labels_with_pbc, np.unique(labels_with_pbc)     # labels, features



    def _apply_pbc(self, labels):
        
        """ 
        Apply periodic boundary conditions to the NxMxL np.array of labeled points.

        Parameters
        ----------

        labels: np.array of NxMxL size
            array of labeles (connected components)
        
        Returns
        ----------
        
        labels, features: np.array of NxMxL size and np.array of its unique labels
            the array returned implies pbc conditions

        """

        faces_left = [labels[-1, :, :],
                      labels[:, -1, :],
                      labels[:, :, -1]
                     ]
        faces_right = [labels[0, :, :],
                       labels[:, 0, :],
                       labels[:, :, 0]
                      ]
        for f1, f2 in zip(faces_left, faces_right):
            for s in np.unique(f1):
                if s == 0:
                    continue
                else:
                    connect = np.unique(f2[f1 == s])
                    for c in connect:
                        if c == 0:
                            continue
                        else:
                            labels[labels == c] = s
        return labels



    def _percolation_dimension(self, labels, features):

        """
        Check percolation dimensionality

        Parameters
        ----------

        labels: np.array
            label from _connected_components method
            
        features: np.array
            label from _connected_components method
        
        Returns
        ----------
        d: dimensionality of percolation
            Note: can be from 2 to 8, which is the number of neighboring unit cells within 3x3x3 supercell
        """

        if len(features) < 1:
            d = 0
        else:
            d = max(Parallel(n_jobs=self.n_jobs,
                backend = self.backend)(delayed(self._percolation_dimension_parallel)(feature, labels) for feature in features))
        return d
    


    def _percolation_dimension_parallel(self, feature, labels):

        if feature == 0:
            d = 0
        else:
            coords = np.argwhere(labels == feature)
            d = self._cross_boundary(coords, np.array(labels.shape)/2)
        return d


    
    def _percolation_energy(self, dim, encut = 10.0):

        """
        Get percolation energy fofr a given dimensionality of percolation

        Parameters
        ----------

        dim: int
            dimensionality of percolation. 2 -> 1D, 4 -> 2D, 8 - 3D percolation
            
        encut: float, 10.0 by default
            stop criterion for the search of percolation energy
        
        Returns
        ----------
        barrier: float
            percolation energy or np.inf if no percolation found
        """
        
        data = self.data.reshape(self.size)
        data = data - data.min()
        emin = data.min()
        emax = emin + encut
        count = 0
        barrier = np.inf
        while (emax - emin) > 0.01:
            count = count + 1
            probe = (emin + emax) / 2
            labels, features = self._connected_components(data, probe)
            if len(features) > 0:
                d = self._percolation_dimension(labels, features)
                if d >= dim:
                    emax = probe
                    barrier = round(emax,4)
                else:
                    emin = probe
            else:
                emin = probe
        return barrier



    def percolation_barriers(self, encut = 10.0, n_jobs = 1, backend = 'threading'):

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
            infromation about percolation {'E_1D': float, 'E_2D': float, 'E_3D': float}

        """

        self.n_jobs = n_jobs
        self.backend = backend

        energies = {}
        for i, dim in enumerate([2, 4, 8]):
            
            energy = self._percolation_energy(encut = encut, dim = dim)
            energies.update({f'E_{i+1}D': energy})
        return energies
    