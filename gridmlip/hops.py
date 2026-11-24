import dataclasses
import json
from pathlib import Path as _FilePath

import numpy as np
from tqdm import tqdm

from ase.io import write
from ions import Edge
from ions.geometry import repeated_supercell

from .grid import Grid, _as_atomic_number
from .energy_map import EnergyMap
from .io import read_cfg, read_energies, write_cfg
from .path import _mic, _write_images

__author__ = "Artem Dembitskiy"


def _supercell_scale(cell, cutoff):
    """Repetitions making the supercell at least `cutoff` thick in every direction (as ions' Edge.superedge)."""
    cell = np.asarray(cell)
    volume = abs(np.linalg.det(cell))
    h1 = volume / np.linalg.norm(np.cross(cell[2, :], cell[1, :]))
    h2 = volume / np.linalg.norm(np.cross(cell[0, :], cell[2, :]))
    h3 = volume / np.linalg.norm(np.cross(cell[0, :], cell[1, :]))
    return tuple(int(n) for n in np.ceil(cutoff / np.array([h1, h2, h3])))


def _segment_distance(points, start, end, cell):
    """Distance of points to the segment start-end under the minimum image convention."""
    d = _mic(np.asarray(points) - start, np.asarray(cell))
    u = end - start
    t = np.clip(d @ u / (u @ u), 0.0, 1.0)
    return np.linalg.norm(d - t[:, None] * u, axis=1)


@dataclasses.dataclass
class Hop:
    """
    Vacancy-assisted hop of a mobile ion, in a supercell: the frame shared by its
    energy landscape, its path and its NEB images.

    Parameters
    ----------
    atoms : ase.Atoms
        Unit cell (with any per-atom arrays the energy model needs). For BVSE without
        oxi_states, the oxidation states are assigned on this cell (no vacancy) and passed
        on to the hop's cells; ``set_oxidation_states`` sets them explicitly.

    source, target : int
        Indices of the hopping ion and of the target site in ``atoms``.

    offset : tuple of 3 ints
        Lattice translation of the target relative to the source, as in the
        edges of ions' Percolator: [source, target, ox, oy, oz].

    supercell_cutoff : float, default 8.0
        The supercell is at least this thick in every direction (Angstrom).

    supercell_size : tuple, optional
        Explicit repetitions (na, nb, nc); overrides supercell_cutoff.

    Attributes
    ----------
    template : ase.Atoms
        Supercell with the target-site ion removed (the vacancy) and the hopping ion on its
        site, with a boolean "moving" array: the first NEB image.

    host : ase.Atoms
        ``template`` without the hopping ion: what the mobile ion sees on the grid.

    moving : int
        Index of the hopping ion in ``template``.

    p0, p1 : np.ndarray
        Cartesian start and end of the hop in the ``template`` frame (unwrapped).

    Example
    -------
    hop = Hop(atoms, 0, 7, (0, 0, -1))
    path = hop.evaluate(gridmlip.BVSE("Na+")).path()
    images = path.images()
    """

    atoms: object
    source: int
    target: int
    offset: tuple
    supercell_cutoff: float = 8.0
    supercell_size: tuple = None

    specie: int = dataclasses.field(init=False)
    scale: tuple = dataclasses.field(init=False)
    supercell: object = dataclasses.field(init=False, repr=False)
    template: object = dataclasses.field(init=False, repr=False)
    host: object = dataclasses.field(init=False, repr=False)
    moving: int = dataclasses.field(init=False)
    p0: np.ndarray = dataclasses.field(init=False, repr=False)
    p1: np.ndarray = dataclasses.field(init=False, repr=False)
    _grids: dict = dataclasses.field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.atoms = self.atoms.copy()
        self.source, self.target = int(self.source), int(self.target)
        self.offset = tuple(int(x) for x in self.offset)
        self.specie = int(self.atoms.numbers[self.source])
        if int(self.atoms.numbers[self.target]) != self.specie:
            raise ValueError(f"source ({self.atoms[self.source].symbol}) and target "
                             f"({self.atoms[self.target].symbol}) are different species")
        if self.supercell_size is not None:
            self.scale = tuple(int(n) for n in self.supercell_size)
        else:
            self.scale = _supercell_scale(self.atoms.cell, self.supercell_cutoff)
        self._build()

    def _build(self):
        """Supercell, template (vacancy + hopping ion) and host from self.atoms (arrays carried over)."""
        self.supercell = repeated_supercell(self.atoms, self.scale)
        edge = Edge(self.supercell, self.source, self.target, np.asarray(self.offset) / np.asarray(self.scale))
        self.template = edge.interpolate(2, center=False)[0]
        self.moving = int(np.flatnonzero(self.template.get_array("moving"))[0])
        self.p0 = self.template.positions[self.moving].copy()
        self.p1 = self.p0 + (edge.p2 - edge.p1)
        self.host = self.template[[k for k in range(len(self.template)) if k != self.moving]]
        self._grids.clear()

    def set_oxidation_states(self, charges):
        """Oxidation states of the unit cell (one per atom of ``atoms``), passed on to the hop's cells."""
        charges = np.asarray(charges, dtype=int)
        if charges.shape != (len(self.atoms),):
            raise ValueError(f"Expected {len(self.atoms)} oxidation states, got {charges.shape}")
        self.atoms.set_array("oxi_states", charges)
        self._build()

    def _prepare_model(self, model):
        """
        BVSE needs oxidation states. Assigned on the vacancy cells they fail (a vacancy breaks the
        charge balance), so they are resolved once on the unit cell and inherited by the hop's cells.
        """
        if not hasattr(model, "resolve_oxidation_states") or "oxi_states" in self.atoms.arrays:
            return
        if isinstance(model.oxidation_states, dict):   # per element: valid for any cell
            return
        self.set_oxidation_states(model.resolve_oxidation_states(self.atoms))

    def __repr__(self):
        symbol = self.atoms[self.source].symbol
        return (f"Hop({symbol}{self.source} -> {symbol}{self.target} {self.offset}, "
                f"length={self.length:.3f}, supercell={self.scale})")

    @property
    def length(self):
        """Hop length (Angstrom)."""
        return float(np.linalg.norm(self.p1 - self.p0))

    def region(self, tube: float = 2.5):
        """
        Points within ``tube`` of the hop segment (a capsule around the hop), as a callable
        for ``Grid.evaluate(region=...)``.
        """
        cell = np.asarray(self.host.cell)
        return lambda points: _segment_distance(points, self.p0, self.p1, cell) <= tube

    def grid(self, resolution: float = 0.2, r_min: float = 0.8):
        """
        Grid of the host (other mobile ions kept, no symmetry reduction), cached per settings.
        """
        key = (resolution, r_min)
        if key not in self._grids:
            self._grids[key] = Grid(self.host, self.specie, resolution=resolution, r_cut=max(5.0, 2 * r_min),
                                    r_min=r_min, symprec=None, empty_framework=False)
        return self._grids[key]

    # ------------------------------------------------------------ evaluation

    def evaluate(self, model, *, resolution: float = 0.2, r_min: float = 0.8, tube: float = 2.5):
        """
        Energy of the mobile ion around the hop.

        Parameters
        ----------
        model:
            as for ``Grid.evaluate``: an ASE calculator, a predictor, or a point model
            such as gridmlip.BVSE

        resolution: float, 0.2 by default
            mesh spacing (Angstrom)

        r_min: float, 0.8 by default
            points closer than this to a host atom are not evaluated (+inf)

        tube: float, 2.5 by default
            only points within this distance of the hop segment are evaluated (Angstrom)

        Returns
        -------
        HopMap
        """
        self._prepare_model(model)
        grid = self.grid(resolution, r_min)
        emap = grid.evaluate(model, supercell_size=(1, 1, 1), region=self.region(tube))
        return HopMap.from_energy_map(emap, self, tube=tube)

    def prepare_configurations(self, *, resolution: float = 0.2, r_min: float = 0.8, tube: float = 2.5,
                               filename: str | None = None, progress: bool = False):
        """
        Host + one mobile ion at each point to evaluate (for external calculations), in the
        order expected by ``load``.

        Returns
        -------
        list of ase.Atoms; atoms.info['grid_index'] is the index of the point
        """
        grid = self.grid(resolution, r_min)
        return grid.prepare_configurations(supercell_size=(1, 1, 1), region=self.region(tube),
                                           filename=filename, progress=progress)

    def load(self, energies, *, resolution: float = 0.2, r_min: float = 0.8, tube: float = 2.5,
             file_format: str | None = None):
        """
        Energies computed elsewhere for ``prepare_configurations()`` (same order and settings).

        Returns
        -------
        HopMap
        """
        grid = self.grid(resolution, r_min)
        emap = grid.load(energies, file_format=file_format, region=self.region(tube))
        return HopMap.from_energy_map(emap, self, tube=tube)


class HopMap(EnergyMap):
    """
    Energy map of a hop: an EnergyMap of the hop's host, +inf outside the evaluated region.

    Example
    -------
    hmap = hop.evaluate(gridmlip.BVSE("Na+"))
    hmap.path().images()
    hmap.write("hop.grd")
    """

    def __init__(self, values, atoms, hop, specie=None, info=None):
        super().__init__(values, atoms, specie=specie, info=info)
        self.hop = hop

    @classmethod
    def from_energy_map(cls, emap, hop, tube=None):
        info = dict(emap.info)
        info["hop"] = {"source": hop.source, "target": hop.target, "offset": list(hop.offset),
                       "supercell": list(hop.scale), "tube": tube}
        return cls(emap.values, emap.atoms, hop, specie=emap.specie, info=info)

    def __repr__(self):
        finite = self.values[np.isfinite(self.values)]
        span = f"{np.ptp(finite):.3f} eV" if finite.size else "empty"
        return f"HopMap({self.hop!r}, shape={self.shape}, range={span}, model={self.info.get('model')!r})"

    def path(self, start=None, end=None, n_images: int = 12, well_radius: float = 0.5, margin: float = 0.0):
        """
        Lowest-barrier path of the hop (see ``EnergyMap.path``); start and end default to
        the hop's sites.

        Returns
        -------
        Path, with ``images()`` and ``write()`` for the NEB
        """
        start = self.hop.p0 if start is None else start
        end = self.hop.p1 if end is None else end
        path = super().path(start, end, n_images=n_images, well_radius=well_radius, margin=margin)
        path.hop = self.hop
        return path


class Hops:
    """
    Vacancy-assisted hops of a percolation network, all in one supercell (so that their
    NEB energies share one reference), in the order of ``edges``.

    Parameters
    ----------
    atoms: ase.Atoms
        unit cell

    specie: int or str
        mobile specie

    edges: (N, 5) int array
        hops [source, target, ox, oy, oz], e.g. ions' ``Percolator.unique_edges(...)``;
        every result keeps this order

    supercell_cutoff: float, 8.0 by default
        the supercell is at least this thick in every direction (Angstrom)

    supercell_size: tuple, optional
        explicit repetitions (na, nb, nc); overrides supercell_cutoff

    resolution, r_min, tube: float, 0.2, 0.8 and 2.5 by default
        mesh spacing, blocking radius and radius of the region evaluated around each hop
        (Angstrom), see ``Hop.evaluate``

    Example
    -------
    pl = Percolator(atoms, "Na", upper_bound=5.0)
    cutoff = pl.cutoff_search(3, 0.5)
    hops = Hops(atoms, "Na", pl.unique_edges(cutoff, 0.5))
    paths = hops.evaluate(gridmlip.BVSE("Na+")).paths()
    paths.write("neb_inputs", format="vasp")
    # NEB -> emins, emaxs in the order of the edges
    pl.find_percolation_barriers(cutoff, 0.5, emins, emaxs, method="naive")
    """

    def __init__(
        self,
        atoms,
        specie,
        edges,
        supercell_cutoff: float = 8.0,
        supercell_size: tuple | None = None,
        resolution: float = 0.2,
        r_min: float = 0.8,
        tube: float = 2.5,
    ):
        self.atoms = atoms.copy()
        self.specie = _as_atomic_number(specie)
        self.edges = np.array(edges, dtype=int).reshape(-1, 5)
        if supercell_size is not None:
            self.scale = tuple(int(n) for n in supercell_size)
        else:
            self.scale = _supercell_scale(self.atoms.cell, supercell_cutoff)
        self.resolution = resolution
        self.r_min = r_min
        self.tube = tube
        self.hops = [Hop(self.atoms, s, t, o, supercell_size=self.scale) for s, t, *o in self.edges]
        wrong = [h for h in self.hops if h.specie != self.specie]
        if wrong:
            raise ValueError(f"Hops of another specie than {self.specie}: {wrong}")

    def __len__(self):
        return len(self.hops)

    def __getitem__(self, i):
        return self.hops[i]

    def __iter__(self):
        return iter(self.hops)

    def __repr__(self):
        return (f"Hops('{self.atoms.symbols}', specie={self.specie}, n_hops={len(self)}, "
                f"supercell={self.scale}, resolution={self.resolution}, r_min={self.r_min}, tube={self.tube})")

    def _settings(self):
        return dict(resolution=self.resolution, r_min=self.r_min, tube=self.tube)

    def summary(self):
        """Print the hops: source -> target, offset, length and points to evaluate."""
        print(f"Hops ({len(self)}), supercell {self.scale}")
        print("────────────────────────────────────────────────────────")
        for i, hop in enumerate(self.hops):
            n = len(hop.grid(self.resolution, self.r_min).region_points(hop.region(self.tube)))
            symbol = self.atoms[hop.source].symbol
            print(f"  {i:3d}  {symbol}{hop.source} -> {symbol}{hop.target} {str(hop.offset):12s} "
                  f"{hop.length:6.3f} A  {n:7,} points")
        print("────────────────────────────────────────────────────────\n")

    # ------------------------------------------------------------ evaluation

    def evaluate(self, model, progress: bool = False):
        """
        Energy maps of all hops (see ``Hop.evaluate``).

        Returns
        -------
        HopMaps, in the order of the edges
        """
        if hasattr(model, "resolve_oxidation_states") and "oxi_states" not in self.atoms.arrays \
                and not isinstance(model.oxidation_states, dict):
            charges = model.resolve_oxidation_states(self.atoms)   # once for all hops, on the unit cell
            self.atoms.set_array("oxi_states", charges)
            for hop in self.hops:
                hop.set_oxidation_states(charges)
        maps = [hop.evaluate(model, **self._settings()) for hop in tqdm(self.hops, desc="hops", disable=not progress)]
        return HopMaps(maps)

    def prepare_configurations(self, filename: str | None = None, progress: bool = False):
        """
        Configurations of all hops for external calculations, hop after hop.

        Parameters
        ----------
        filename: str, optional
            also write the configurations (.cfg for MLIP 2.0, anything else via ASE)

        progress: bool, False by default
            show a progress bar

        Returns
        -------
        list of ase.Atoms; atoms.info['hop_index'] and atoms.info['grid_index']
        """
        configurations = []
        for i, hop in enumerate(tqdm(self.hops, desc="hops", disable=not progress)):
            for cfg in hop.prepare_configurations(**self._settings()):
                cfg.info["hop_index"] = i
                configurations.append(cfg)
        if filename:
            if str(filename).endswith(".cfg"):
                write_cfg(filename, configurations)
            else:
                write(filename, configurations)
        return configurations

    def load(self, energies, file_format: str | None = None):
        """
        Energies computed elsewhere for ``prepare_configurations()`` (same order).

        Parameters
        ----------
        energies: array-like, str or Path
            energies (eV) of all configurations, or a file with the processed configurations

        file_format: str, optional
            'cfg' to force the MLIP 2.0 reader; otherwise inferred from the extension

        Returns
        -------
        HopMaps
        """
        if isinstance(energies, (str, bytes)) or hasattr(energies, "__fspath__"):
            source = str(energies)
            if file_format == "cfg" and not source.endswith(".cfg"):
                energies = np.array([a.get_potential_energy() for a in read_cfg(source)])
            else:
                energies = read_energies(source)
        energies = np.asarray(energies, dtype=float).ravel()
        counts = [len(h.grid(self.resolution, self.r_min).region_points(h.region(self.tube))) for h in self.hops]
        if energies.size != sum(counts):
            raise ValueError(f"Expected {sum(counts)} energies (one per configuration), got {energies.size}")
        chunks = np.split(energies, np.cumsum(counts)[:-1])
        return HopMaps([hop.load(e, **self._settings()) for hop, e in zip(self.hops, chunks)])


class HopMaps:
    """Energy maps of ``Hops`` (one HopMap per hop, in the order of the edges)."""

    def __init__(self, maps):
        self.maps = list(maps)

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, i):
        return self.maps[i]

    def __iter__(self):
        return iter(self.maps)

    def __repr__(self):
        return f"HopMaps(n_hops={len(self)})"

    def paths(self, n_images: int = 12, well_radius: float = 0.5, margin: float = 0.0):
        """
        Lowest-barrier path of every hop (see ``HopMap.path``).

        Returns
        -------
        Paths, in the order of the edges
        """
        return Paths([m.path(n_images=n_images, well_radius=well_radius, margin=margin) for m in self.maps])

    def save(self, folder):
        """Save the maps as hop_000.npz, hop_001.npz, ... (reload with HopMaps.load)."""
        folder = _FilePath(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.maps):
            m.save(folder / f"hop_{i:03d}.npz")

    @classmethod
    def load(cls, folder, hops):
        """Maps saved with HopMaps.save, attached to the given Hops (same order)."""
        folder = _FilePath(folder)
        maps = []
        for i, hop in enumerate(hops):
            emap = EnergyMap.load(folder / f"hop_{i:03d}.npz")
            maps.append(HopMap(emap.values, emap.atoms, hop, specie=emap.specie, info=emap.info))
        return cls(maps)


class Paths:
    """
    NEB starting paths of ``Hops`` (one Path per hop, in the order of the edges).

    Example
    -------
    paths = hops.evaluate(model).paths(n_images=12)
    paths.barriers          # grid estimates
    paths.images()          # images()[i] belongs to edges[i]
    paths.write("neb_inputs", format="vasp")
    """

    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.paths[i]

    def __iter__(self):
        return iter(self.paths)

    def __repr__(self):
        return f"Paths(n_hops={len(self)}, barriers={np.round(self.barriers, 3).tolist()})"

    @property
    def barriers(self):
        """Grid barriers of the hops (eV), np.ndarray in the order of the edges."""
        return np.array([p.barrier for p in self.paths])

    def images(self):
        """NEB images of every hop: list (per hop) of lists of ase.Atoms."""
        return [p.images() for p in self.paths]

    def write(self, folder, format: str = "extxyz"):
        """
        Write the NEB images of every hop and a summary (paths.json).

        Parameters
        ----------
        folder: str or Path
            output folder: hop_000.xyz, ... for 'extxyz'; hop_000/00/POSCAR, ... for 'vasp'

        format: str, 'extxyz' by default
            'extxyz' or 'vasp'
        """
        folder = _FilePath(folder)
        folder.mkdir(parents=True, exist_ok=True)
        summary = []
        for i, p in enumerate(self.paths):
            name = f"hop_{i:03d}.xyz" if format == "extxyz" else f"hop_{i:03d}"
            _write_images(p.images(), folder / name, format)
            hop = p.hop
            summary.append({"hop": i, "file": name, "source": hop.source, "target": hop.target,
                            "offset": list(hop.offset), "supercell": list(hop.scale),
                            "length": hop.length, "barrier": p.barrier, "level": p.level})
        (folder / "paths.json").write_text(json.dumps(summary, indent=1))
