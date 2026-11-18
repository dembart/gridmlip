import dataclasses
import re
import warnings

import numpy as np
from tqdm import tqdm

from ase.data import atomic_numbers
from ase.io import read, write

from .io import read_cfg, write_cfg, read_energies
from ._neighborhood import nn_list
from ._mesh import Mesh
from .energy_map import EnergyMap

__author__ = "Artem Dembitskiy"


def _deprecated(old, new):
    warnings.warn(f"{old} is deprecated and will be removed in gridmlip 0.4; use {new} instead.",
                  DeprecationWarning, stacklevel=3)


def _as_atomic_number(specie):
    if isinstance(specie, str):
        symbol = re.sub(r"[\d+\-]", "", specie)
        return atomic_numbers[symbol]
    return int(specie)


@dataclasses.dataclass
class Grid:
    """
    Symmetry-aware mesh of a unit cell: where to place the mobile ion.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure.

    specie : int or str
        Mobile specie, as atomic number (3) or symbol ("Li").

    resolution : float, default 0.25
        Approximate spacing between points (Angstrom). The number of points per axis may be
        rounded up so that the mesh is commensurate with the symmetry operations.

    r_cut : float, default 5.0
        Cutoff for the nearest-neighbor search and default supercell size
        of the configurations (see ``prepare_configurations``).

    r_min : float, default 1.0
        Blocking sphere radius: points closer than this to the host are not evaluated (+inf).

    symprec : float or None, default 0.1
        Symmetry precision (Angstrom). None disables symmetry reduction.

    atomic_types_mapper : dict, optional
        Mapper of atomic numbers into species used by the MLIP.

    empty_framework : bool, default True
        Remove the other mobile ions from the host before sampling.

    verbose : bool, default False
        Print a mesh summary when the mesh is generated.

    Example
    -------
    grid = Grid(atoms, specie=3, resolution=0.2)
    emap = grid.evaluate(calc)
    emap.percolation_barriers()
    """

    atoms: object
    specie: int
    resolution: float = 0.25
    r_cut: float = 5.0
    r_min: float = 1.0
    symprec: float = 0.1
    atomic_types_mapper: dict = None
    empty_framework: bool = True
    verbose: bool = False

    cell: object = dataclasses.field(init=False)
    base: object = dataclasses.field(init=False)
    _mesh: object = dataclasses.field(default=None, init=False)

    def __post_init__(self):

        assert self.r_cut > self.r_min, f"r_cut ({self.r_cut}) must be greater than r_min ({self.r_min})"

        self.specie = _as_atomic_number(self.specie)
        self.atoms = self.atoms.copy()
        if self.atomic_types_mapper:
            self.atoms.numbers = self._map_atomic_types(self.atomic_types_mapper, self.atoms.numbers)
            self.specie = self.atomic_types_mapper[self.specie]
        self.cell = self.atoms.cell
        self._host_mask = (self.atoms.numbers != self.specie) if self.empty_framework \
            else np.ones(len(self.atoms), dtype=bool)
        self.base = self.atoms[self._host_mask]
        self._reductions = {}

    def __repr__(self):
        return (f"Grid(atoms='{self.atoms.symbols}', specie={self.specie}, "
                f"resolution={self.resolution}, r_cut={self.r_cut}, "
                f"r_min={self.r_min}, symprec={self.symprec}, "
                f"empty_framework={self.empty_framework})")

    # ------------------------------------------------------------------ mesh

    def generate_mesh(self):
        """Build the mesh, its irreducible points and distances to the host (done automatically)."""
        self._mesh = Mesh.from_atoms(self.base, resolution=self.resolution, symprec=self.symprec)
        self.mesh_shape = self._mesh.mesh_shape
        self.ir_grid_points, self.mapping = self._mesh.get_irreducible_mapping()
        self.mesh_frac = self._mesh.frac_coords
        self.mesh_cart = np.dot(self.mesh_frac, self.base.cell)
        self.min_dists, _ = nn_list(self.base.positions,
                                    self.mesh_cart[self.ir_grid_points],
                                    self.r_cut,
                                    self.cell)
        self._reductions = {None: (self.ir_grid_points, self.mapping, self.min_dists)}
        if self.verbose:
            self.summary()

    def _ensure_mesh(self):
        if self._mesh is None:
            self.generate_mesh()

    def summary(self):
        """Print the mesh summary."""
        self._ensure_mesh()
        total_points = self.mesh_frac.shape[0]
        irreducible_points = self.ir_grid_points.shape[0]
        passed_filter = len(self)
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

    def _ops_preserving(self, labels):
        """Mesh symmetry operations that map every host atom onto an atom with the same label."""
        labels = np.asarray(labels)
        frac = self.base.get_scaled_positions(wrap=True)
        cell = np.asarray(self.base.cell)
        tol = max(self.symprec or 0.0, 1e-3)
        n = np.asarray(self.mesh_shape)
        kept_R, kept_t = [], []
        for R, t_idx in zip(self._mesh.rotations, self._mesh.translations):
            t = np.asarray(t_idx) / n
            img = frac @ np.asarray(R).T + t
            d = img[:, None, :] - frac[None, :, :]
            d -= np.rint(d)
            dist = np.linalg.norm(d @ cell, axis=-1)
            j = dist.argmin(axis=1)
            if np.all(dist[np.arange(len(frac)), j] < tol) and np.all(labels[j] == labels):
                kept_R.append(R)
                kept_t.append(t)
        return kept_R, kept_t

    def _reduction(self, labels=None):
        """
        (ir_grid_points, mapping, min_dists) for the symmetry that also preserves per-host-atom labels.
        labels=None means the symmetry of the host as given.
        """
        self._ensure_mesh()
        key = None if labels is None else tuple(np.asarray(labels).tolist())
        if key in self._reductions:
            return self._reductions[key]
        kept_R, kept_t = self._ops_preserving(labels)
        if len(kept_R) == len(self._mesh.rotations):
            red = self._reductions[None]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mesh = Mesh(self.mesh_shape, kept_R, kept_t)
            ir, mapping = mesh.get_irreducible_mapping()
            full_min = self.min_dists[self.mapping]
            red = (ir, mapping, full_min[ir])
        self._reductions[key] = red
        return red

    @property
    def points(self):
        """Cartesian coordinates of the irreducible points that are not blocked (to be evaluated)."""
        ir, _, min_dists = self._reduction()
        return self.mesh_cart[ir][min_dists > self.r_min]

    def __len__(self):
        """Number of points to evaluate."""
        _, _, min_dists = self._reduction()
        return int((min_dists > self.r_min).sum())

    @property
    def min_distances(self):
        """Distance from every mesh point to the nearest host atom, as a 3D array."""
        self._ensure_mesh()
        return self.min_dists[self.mapping].reshape(self.mesh_shape)

    # kept for backward compatibility
    min_distances_data = min_distances

    def percolation_radii(self, backend: str = "auto"):
        """
        Largest radius of a sphere that percolates through the host in 1, 2 and 3 dimensions.

        Returns
        -------
        dict {'r1d': float, 'r2d': float, 'r3d': float}
        """
        from ._percolation_analysis import Percolyze
        return Percolyze(self.min_distances).percolation_radii(backend=backend)

    # ------------------------------------------------------- configurations

    def _map_atomic_types(self, atomic_types_mapper: dict, numbers):
        u, inv = np.unique(numbers, return_inverse=True)
        return np.array([atomic_types_mapper[x] for x in u])[inv].reshape(numbers.shape)

    @staticmethod
    def _repeat_atoms(atoms, r_cut, supercell_size=None):
        atoms = atoms.copy()

        if supercell_size is not None:
            return atoms.repeat(supercell_size)

        cell = atoms.cell
        h1 = cell.volume/np.linalg.norm(np.cross(cell[2, :], cell[1, :]))
        h2 = cell.volume/np.linalg.norm(np.cross(cell[0, :], cell[2, :]))
        h3 = cell.volume/np.linalg.norm(np.cross(cell[0, :], cell[1, :]))
        scale = np.array(np.ceil(r_cut/np.array([h1, h2, h3])), dtype=int)
        return atoms.repeat(scale)

    def prepare_configurations(
        self,
        supercell_cutoff: float | None = None,
        supercell_size: tuple | None = None,
        filename: str | None = None,
        config_format: str = "ase",
        progress: bool = False,
    ):
        """
        Host supercell + one mobile ion at each point of ``grid.points`` (same order).

        Parameters
        ----------
        supercell_cutoff: float, optional
            the supercell is at least this thick in every direction (Angstrom); defaults to r_cut

        supercell_size: tuple, optional
            explicit repetitions (na, nb, nc); overrides supercell_cutoff

        filename: str, optional
            also write the configurations (.cfg for MLIP 2.0, anything else via ASE)

        config_format: str, 'ase' by default
            'ase' (ase.Atoms) or 'pymatgen' (Structure)

        progress: bool, False by default
            show a progress bar

        Returns
        -------
        list of configurations; atoms.info['grid_index'] is the index in grid.points
        """
        points = self.points
        cutoff = self.r_cut if supercell_cutoff is None else supercell_cutoff
        base = self._repeat_atoms(self.base, cutoff, supercell_size=supercell_size)
        configurations = []

        if config_format == "ase":
            for i, p in enumerate(tqdm(points, desc="creating configurations", disable=not progress)):
                framework = base.copy()
                framework.append(self.specie)
                framework.positions[-1] = p
                framework.info["grid_index"] = i
                configurations.append(framework)
        elif config_format == "pymatgen":
            from pymatgen.io.ase import AseAtomsAdaptor
            base = AseAtomsAdaptor.get_structure(base)
            for p in tqdm(points, desc="creating configurations", disable=not progress):
                framework = base.copy()
                framework.append(self.specie, coords=p, coords_are_cartesian=True)
                configurations.append(framework)
        else:
            raise ValueError(f"Wrong config_format {config_format}")

        if filename:
            if config_format != "ase":
                raise ValueError('Only "ase" format is allowed for saving files')
            if str(filename).endswith(".cfg"):
                write_cfg(filename, configurations)
            else:
                write(filename, configurations)
        return configurations

    # ------------------------------------------------------------ evaluation

    def evaluate(self, model, *, supercell_cutoff: float | None = None, supercell_size: tuple | None = None):
        """
        Evaluate a model on the grid.

        Parameters
        ----------
        model:
            - an ASE calculator (MACE and SevenNet calculators are batched automatically)
            - a predictor from gridmlip.predictors, or any callable list[Atoms] -> energies
            - a point model such as gridmlip.BVSE

        supercell_cutoff, supercell_size:
            supercell of the configurations (structure models only), see prepare_configurations

        Returns
        -------
        EnergyMap
        """
        from .models.base import PointModel
        from .predictors import as_predictor

        if isinstance(model, PointModel):
            host = model.prepare(self.atoms, self._host_mask)
            labels = model.symmetry_labels(host)
            reduction = self._reduction(labels)
            ir, _, min_dists = reduction
            points = self.mesh_cart[ir][min_dists > self.r_min]
            energies = model(points, host)
            info = model.describe(host)
        else:
            predictor = as_predictor(model)
            cfgs = self.prepare_configurations(supercell_cutoff=supercell_cutoff, supercell_size=supercell_size)
            energies = predictor(cfgs)
            reduction = self._reduction()
            info = dict(getattr(predictor, "info", {}) or {"model": repr(model)})
        return self._energy_map(energies, reduction, info)

    def load_energies(self, energies, file_format: str | None = None):
        """
        Energies computed elsewhere for ``prepare_configurations()`` (same order).

        Parameters
        ----------
        energies: array-like or str
            energies (eV) or a file with processed configurations (.cfg, .xyz, ...)

        file_format: str, optional
            'cfg' to force the MLIP 2.0 reader; otherwise inferred from the extension

        Returns
        -------
        EnergyMap
        """
        source = None
        if isinstance(energies, (str, bytes)) or hasattr(energies, "__fspath__"):
            source = str(energies)
            if file_format == "cfg" and not source.endswith(".cfg"):
                energies = np.array([a.get_potential_energy() for a in read_cfg(source)])
            else:
                energies = read_energies(source)
        info = {"model": "external", "source": source} if source else {"model": "external"}
        emap = self._energy_map(energies, self._reduction(), info)

        # backward compatibility: grid.data / grid.percolation_barriers() keep working until 0.4
        self.energies = np.asarray(energies, dtype=float).ravel()
        self.distribution = emap.values.ravel()
        self.data = emap.values
        self._legacy_map = emap
        return emap

    def _energy_map(self, energies, reduction, info):
        ir, mapping, min_dists = reduction
        accessible = min_dists > self.r_min
        energies = np.asarray(energies, dtype=float).ravel()
        if energies.size != accessible.sum():
            raise ValueError(
                f"Expected {accessible.sum()} energies (one per configuration), got {energies.size}"
            )
        # blocked points and failed predictions are +inf
        ir_values = np.full(min_dists.shape, np.inf)
        ir_values[accessible] = np.where(np.isnan(energies), np.inf, energies)
        values = ir_values[mapping].reshape(self.mesh_shape)

        info = dict(info)
        info.setdefault("grid", {
            "specie": int(self.specie), "resolution": self.resolution, "r_min": self.r_min,
            "r_cut": self.r_cut, "symprec": self.symprec, "empty_framework": self.empty_framework,
            "mesh_shape": list(self.mesh_shape), "n_points": int(accessible.sum()),
        })
        return EnergyMap(values, self.atoms, specie=self.specie, info=info)

    # ------------------------------------------------- deprecated (until 0.4)

    @classmethod
    def from_file(cls, file: str, specie, **kwargs):
        """Create a Grid from a structure file (.cfg of MLIP 2.0, or anything ASE reads)."""
        atoms = read_cfg(file)[0] if str(file).endswith(".cfg") else read(file)
        return cls(atoms, specie, **kwargs)

    def construct_cfgs(self, config_format: str = "ase", filename: str | None = None,
                       supercell_size: tuple | None = None):
        """Deprecated: use prepare_configurations()."""
        _deprecated("Grid.construct_cfgs()", "Grid.prepare_configurations()")
        return self.prepare_configurations(supercell_size=supercell_size, filename=filename,
                                           config_format=config_format, progress=True)

    def read_processed_cfgs(self, filename: str, file_format: str = "xyz"):
        """Deprecated: use load_energies(filename)."""
        _deprecated("Grid.read_processed_cfgs()", "Grid.load_energies(filename)")
        return self.load_energies(filename, file_format=file_format)

    def _legacy(self, old, new):
        _deprecated(f"Grid.{old}()", new)
        if getattr(self, "_legacy_map", None) is None:
            raise RuntimeError("No energies loaded. Call load_energies() first.")
        return self._legacy_map

    def percolation_barriers(self, upper_limit: float = 10.0, backend: str = "auto"):
        """Deprecated: use EnergyMap.percolation_barriers()."""
        return self._legacy("percolation_barriers", "EnergyMap.percolation_barriers()") \
            .percolation_barriers(upper_limit=upper_limit, backend=backend)

    def write_grd(self, filename: str):
        """Deprecated: use EnergyMap.write('name.grd')."""
        self._legacy("write_grd", "EnergyMap.write('name.grd')").write(f"{filename}.grd")

    def write_cube(self, filename: str):
        """Deprecated: use EnergyMap.write('name.cube')."""
        self._legacy("write_cube", "EnergyMap.write('name.cube')").write(f"{filename}.cube")
