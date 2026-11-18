import json

import numpy as np
from ase import Atoms

from ._percolation_analysis import Percolyze
from ._void_search import clustered_local_minima
from .io import write_cube, write_grd

__author__ = "Artem Dembitskiy"


class EnergyMap:
    """
    Energies of a mobile ion on a periodic mesh, as returned by
    ``Grid.evaluate()`` and ``Grid.load_energies()``.

    Point (i, j, k) of ``values`` sits at fractional coordinates (i/nx, j/ny, k/nz).
    Blocked points (closer than ``r_min`` to the host) are ``+inf``.

    Example
    -------

    emap = grid.evaluate(calc)
    emap.percolation_barriers()   # {'e1d': ..., 'e2d': ..., 'e3d': ...}
    emap.write('map.grd')
    """

    def __init__(self, values, atoms, specie=None, info=None):
        """
        Parameters
        ----------
        values: np.ndarray of shape (nx, ny, nz)
            energies on the periodic mesh (eV), +inf for blocked points

        atoms: ase.Atoms
            structure the map belongs to (defines the cell, written to .cube files)

        specie: int, optional
            atomic number of the mobile specie

        info: dict, optional
            provenance: model, settings, oxidation states used, ...
        """
        values = np.asarray(values, dtype=float)
        if values.ndim != 3:
            raise ValueError(f"values must be a 3D array, got shape {values.shape}")
        self.values = values
        self.atoms = atoms.copy()
        self.atoms.calc = None
        self.specie = specie
        self.info = dict(info or {})

    @property
    def cell(self):
        return self.atoms.cell

    @property
    def shape(self):
        return self.values.shape

    def __repr__(self):
        finite = self.values[np.isfinite(self.values)]
        span = f"{np.ptp(finite):.3f} eV" if finite.size else "empty"
        model = self.info.get("model", "unknown model")
        return (f"EnergyMap({self.atoms.get_chemical_formula()}, specie={self.specie}, "
                f"shape={self.shape}, range={span}, model={model!r})")

    def percolation_barriers(self, upper_limit: float = 10.0, backend: str = "auto"):
        """
        Lowest energies (relative to the map minimum) at which the accessible region
        percolates in 1, 2 and 3 dimensions.

        Parameters
        ----------
        upper_limit: float, 10.0 by default
            barriers above this energy are reported as np.inf

        backend: str, 'auto' by default
            'numba' or 'scipy'; both give identical results, numba is 10-100 times faster

        Returns
        -------
        dict {'e1d': float, 'e2d': float, 'e3d': float}
        """
        return Percolyze(self.values).percolation_barriers(upper_limit=upper_limit, backend=backend)

    def local_minima(self, min_distance: float = 2.0, energy_window: float = 1.0, coord_type: str = "cart"):
        """
        Local energy minima, clustered so that no two are closer than ``min_distance``.

        Parameters
        ----------
        min_distance: float, 2.0 by default
            minimal distance between reported minima (Angstrom)

        energy_window: float, 1.0 by default
            only minima within this energy above the global minimum are reported (eV)

        coord_type: str, 'cart' by default
            'cart', 'frac' or 'mesh' (integer mesh indices)

        Returns
        -------
        list of (position, energy, n_points) sorted by energy
        """
        return clustered_local_minima(
            self.values,
            min_distance=min_distance,
            relative_max_value=energy_window,
            cell=np.asarray(self.cell),
            coord_type=coord_type,
        )

    def write(self, filename: str):
        """Write the map to .grd (VESTA) or .cube; the format is taken from the extension."""
        filename = str(filename)
        if filename.endswith(".grd"):
            write_grd(filename, self.values, self.cell)
        elif filename.endswith(".cube"):
            write_cube(filename, self.values, self.atoms)
        else:
            raise ValueError(f"Unknown format for {filename!r}: use a .grd or .cube extension")

    def save(self, filename: str):
        """Save the map and its structure to .npz (reload with EnergyMap.load)."""
        np.savez_compressed(
            filename,
            values=self.values,
            numbers=self.atoms.numbers,
            positions=self.atoms.positions,
            cell=np.asarray(self.cell),
            pbc=self.atoms.pbc,
            specie=-1 if self.specie is None else int(self.specie),
            info=json.dumps(self.info, default=str),
        )

    @classmethod
    def load(cls, filename: str):
        """Load a map saved with EnergyMap.save."""
        with np.load(filename, allow_pickle=False) as f:
            atoms = Atoms(numbers=f["numbers"], positions=f["positions"], cell=f["cell"], pbc=f["pbc"])
            specie = int(f["specie"])
            return cls(
                f["values"],
                atoms,
                specie=None if specie < 0 else specie,
                info=json.loads(str(f["info"])),
            )
