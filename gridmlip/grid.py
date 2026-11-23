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


_LABEL_WORDS = ("energy", "force", "stress", "virial", "dipole", "charges")


def _strip_labels(atoms):
    """
    Remove reference labels inherited from the parent structure (energy, forces, stress, ...)
    from atoms.info and atoms.arrays, so generated configurations are never mistaken for labeled data.
    """
    for key in [k for k in atoms.info if k.startswith("REF_") or any(w in k.lower() for w in _LABEL_WORDS)]:
        del atoms.info[key]
    for key in [k for k in atoms.arrays if k.startswith("REF_") or any(w in k.lower() for w in _LABEL_WORDS)]:
        del atoms.arrays[key]
    atoms.calc = None
    return atoms


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

    def _generate_mesh(self):
        """Build the mesh, its irreducible points and distances to the host (on first use)."""
        self._mesh = Mesh.from_atoms(self.base, resolution=self.resolution, symprec=self.symprec)
        self._ir_grid_points, self._mapping = self._mesh.get_irreducible_mapping()
        self._mesh_frac = self._mesh.frac_coords
        self._mesh_cart = np.dot(self._mesh_frac, self.base.cell)
        self._min_dists, _ = nn_list(self.base.positions,
                                    self._mesh_cart[self._ir_grid_points],
                                    self.r_cut,
                                    self.cell)
        self._reductions = {None: (self._ir_grid_points, self._mapping, self._min_dists)}
        if self.verbose:
            self.summary()

    def _ensure_mesh(self):
        if self._mesh is None:
            self._generate_mesh()

    @property
    def mesh_shape(self):
        """Number of mesh points along each cell vector."""
        self._ensure_mesh()
        return self._mesh.mesh_shape

    def summary(self):
        """Print the mesh summary."""
        self._ensure_mesh()
        total_points = self._mesh_frac.shape[0]
        irreducible_points = self._ir_grid_points.shape[0]
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
            full_min = self._min_dists[self._mapping]
            red = (ir, mapping, full_min[ir])
        self._reductions[key] = red
        return red

    def _selection(self, reduction, region=None):
        """Irreducible points to evaluate: not blocked and, if a region is given, inside it."""
        ir, mapping, min_dists = reduction
        selected = min_dists > self.r_min
        if region is None:
            return selected
        inside = region(self._mesh_cart) if callable(region) else region
        inside = np.asarray(inside, dtype=bool).ravel()
        if inside.size != len(mapping):
            raise ValueError(f"region must cover the mesh: expected {len(mapping)} values, got {inside.size}")
        # an irreducible point is evaluated if any of its images lies inside the region
        ir_inside = np.zeros(len(ir), dtype=bool)
        ir_inside[mapping[inside]] = True
        return selected & ir_inside

    @property
    def points(self):
        """Cartesian coordinates of the irreducible points that are not blocked (to be evaluated)."""
        return self.region_points()

    def region_points(self, region=None):
        """
        Cartesian coordinates of the irreducible points to evaluate inside a region.

        Parameters
        ----------
        region: optional
            boolean array of shape ``mesh_shape``, or a callable taking Cartesian
            positions (N, 3) of the mesh points and returning a boolean array (N,);
            None means the whole mesh

        Returns
        -------
        np.ndarray of shape (n, 3)
        """
        reduction = self._reduction()
        ir, _, _ = reduction
        return self._mesh_cart[ir][self._selection(reduction, region)]

    def __len__(self):
        """Number of points to evaluate."""
        _, _, min_dists = self._reduction()
        return int((min_dists > self.r_min).sum())

    @property
    def min_distances(self):
        """Distance from every mesh point to the nearest host atom, as a 3D array."""
        self._ensure_mesh()
        return self._min_dists[self._mapping].reshape(self.mesh_shape)

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
        region=None,
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

        region: optional
            only points inside this region (see ``region_points``); the order is that
            of ``grid.region_points(region)``

        Returns
        -------
        list of configurations; atoms.info['grid_index'] is the index in grid.points
        (in grid.region_points(region) with a region)
        """
        points = self.region_points(region)
        cutoff = self.r_cut if supercell_cutoff is None else supercell_cutoff
        base = _strip_labels(self._repeat_atoms(self.base, cutoff, supercell_size=supercell_size))
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

    def evaluate(self, model, *, supercell_cutoff: float | None = None, supercell_size: tuple | None = None,
                 region=None):
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

        region: optional
            evaluate only the points inside this region, the others are +inf: a boolean
            array of shape ``mesh_shape`` or a callable of Cartesian positions (see
            ``region_points``). Saves evaluations when only part of the cell matters,
            e.g. the surroundings of one hop

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
            ir, _, _ = reduction
            points = self._mesh_cart[ir][self._selection(reduction, region)]
            energies = model(points, host)
            info = model.describe(host)
        else:
            predictor = as_predictor(model)
            cfgs = self.prepare_configurations(supercell_cutoff=supercell_cutoff, supercell_size=supercell_size,
                                               region=region)
            energies = predictor(cfgs)
            reduction = self._reduction()
            info = dict(getattr(predictor, "info", {}) or {"model": repr(model)})
        return self._energy_map(energies, reduction, info, region)

    def load(self, energies, file_format: str | None = None, region=None):
        """
        Energies computed elsewhere for ``prepare_configurations()`` (same order).

        Parameters
        ----------
        energies: array-like, str or Path
            a list/array of energies (eV), or a file with the processed configurations
            (.cfg of MLIP 2.0, or any format ASE reads with energies, e.g. .xyz)

        file_format: str, optional
            'cfg' to force the MLIP 2.0 reader; otherwise inferred from the extension

        region: optional
            the region used in ``prepare_configurations(region=...)``

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
        return self._energy_map(energies, self._reduction(), info, region)

    def _energy_map(self, energies, reduction, info, region=None):
        ir, mapping, min_dists = reduction
        accessible = self._selection(reduction, region)
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
            "region": region is not None,
        })
        return EnergyMap(values, self.atoms, specie=self.specie, info=info)

    # ------------------------------------------------------------- loading

    @classmethod
    def from_file(cls, file: str, specie, **kwargs):
        """Create a Grid from a structure file (.cfg of MLIP 2.0, or anything ASE reads)."""
        atoms = read_cfg(file)[0] if str(file).endswith(".cfg") else read(file)
        return cls(atoms, specie, **kwargs)
