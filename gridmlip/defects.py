import numpy as np
from tqdm import tqdm

from spglib import get_symmetry_dataset
from ions import Percolator, Edge

from .core import Grid, _strip_labels
from ._neighborhood import nn_list



_DEFECT_NAMES = {"int": "interstitial", "vac": "vacancy", "frenkel": "frenkel_pair"}


class _DefectSampler:
    """Shared machinery of interstitials(), vacancies() and frenkel_pairs()."""

    def __init__(
        self,
        atoms,
        specie: int,
        resolution: float = 0.25,
        r_cut: float = 6.0,
        r_min: float = 1.0,
        symprec: float | None = 0.1,
        max_sep_dist: float | str = 'auto',
        upper_bound: float = 6.0,
        verbose: bool = False,
    ):
        if specie is None:
            raise ValueError("specie is required (atomic number of the mobile species)")

        self.atoms = atoms.copy()
        self.specie = specie
        self.resolution = resolution
        self.r_cut = r_cut
        self.r_min = r_min
        self.symprec = symprec
        self.max_sep_dist = max_sep_dist
        self.verbose = verbose
        self.upper_bound = upper_bound
        self._specie_symbol = atoms[atoms.numbers == specie].symbols[0]
        self._percolator = None # placeholder
        self._mincut_maxdim = None # placeholder


    def _make_grid(self, atoms):
        grid = Grid(
            atoms,
            specie=self.specie,
            resolution=self.resolution,
            symprec=self.symprec,
            empty_framework=False,
            r_cut=self.r_cut,
            r_min=self.r_min,
            verbose=self.verbose,
        )
        grid.generate_mesh()
        return grid

    @staticmethod
    def _valid_positions(mesh_pos, framework, r_min):
        """Filter candidate positions using the actual framework."""
        if r_min <= 0:
            return mesh_pos

        distances, _ = nn_list(framework.positions, mesh_pos, r_cut=r_min, cell=framework.cell)

        return mesh_pos[distances >= r_min]


    @staticmethod
    def _remove_atoms(atoms, indices):
        mask = np.ones(len(atoms), dtype=bool)
        mask[indices] = False
        return atoms[mask].copy()


    @staticmethod
    def _filter_by_distance(mesh_pos, center, r_cut, cell):
        """Keep points within cutoff of center."""
        distances, _ = nn_list(center.reshape(1, 3), mesh_pos, r_cut=r_cut, cell=cell)
        return mesh_pos[distances <= r_cut]


    def _get_percolator(self):
        """Note: bottleneck_radius is hardcoded to 0.0 here."""
        if self._percolator is None:
            pl = Percolator(self.atoms, mobile_specie=self._specie_symbol, upper_bound=self.upper_bound)
            mincut, maxdim = pl.mincut_maxdim(bottleneck_radius=0.0)
            self._percolator = pl
            self._mincut_maxdim = (mincut, maxdim)
        return (self._percolator, *self._mincut_maxdim)


    def _prepare_site(self, atoms, base, remove_ids, center, max_sep):
        grid = self._make_grid(self._remove_atoms(atoms, remove_ids))
        filtered_pos = grid.points
        filtered_pos = self._filter_by_distance(filtered_pos, center, max_sep, atoms.cell)
        
        # filtering due to the lower defect concentration in a supercell
        defect_base = self._remove_atoms(base, remove_ids)
        
        filtered_pos = self._valid_positions(
            filtered_pos,
            defect_base[defect_base.numbers == self.specie],
            self.r_min,
        )
        
        return filtered_pos, defect_base


    def _prepare_interstitials(self, supercell_size=None):

        atoms = self.atoms.copy()
        grid = self._make_grid(atoms)
        base = grid._repeat_atoms(atoms, grid.r_cut, supercell_size=supercell_size)
        filtered_pos = grid.points
        return [(filtered_pos, base)]



    def _prepare_vacancies(self, supercell_size=None):

        atoms = self.atoms.copy()
        pl, mincut, maxdim = self._get_percolator()
        edges = pl.unique_edges(cutoff=mincut, bottleneck_radius=0.0)

        base = Grid._repeat_atoms(atoms, self.r_cut, supercell_size=supercell_size)

        sites_and_base = []
        for edge_data in edges:

            source, target, offset = (edge_data[0], edge_data[1], edge_data[2:])
            scale = (atoms.cell.cellpar()[:3] / base.cell.cellpar()[:3])
            edge = Edge(base, source, target, offset * scale)

            local_max_sep = self.max_sep_dist
            if local_max_sep == "auto":
                local_max_sep = edge.length / 2 + self.resolution

            filtered_pos, defect_base = self._prepare_site(
                atoms, base, [source, target], edge.centroid, local_max_sep
            )
            sites_and_base.append((filtered_pos, defect_base))

        return sites_and_base



    def _prepare_frenkels(self, supercell_size=None):

        atoms = self.atoms.copy()

        equivalent_atoms = np.arange(len(atoms))
        if self.symprec is not None:
            dataset = get_symmetry_dataset(
                (
                    atoms.cell,
                    atoms.get_scaled_positions(),
                    atoms.numbers,
                ),
                symprec=self.symprec,
            )
            equivalent_atoms = dataset.equivalent_atoms

        ineq_sites = np.unique(equivalent_atoms)
        vac_ids = ineq_sites[atoms.numbers[ineq_sites] == self.specie]

        if self.max_sep_dist == "auto":
            _, mincut, _ = self._get_percolator()
            global_max_sep = mincut
        else:
            global_max_sep = self.max_sep_dist

        base = Grid._repeat_atoms(atoms, self.r_cut, supercell_size=supercell_size)

        sites_and_base = []
        for vac_id in vac_ids:
            filtered_pos, defect_base = self._prepare_site(
                atoms, base, [vac_id], atoms.positions[vac_id], global_max_sep
            )
            sites_and_base.append((filtered_pos, defect_base))
        return sites_and_base


    @staticmethod
    def _statistics(sites_and_base):
        summary = {
            'points': sum([d[0].shape[0] for d in sites_and_base]),
            'bases': len(sites_and_base),
        }
        return summary


    def _append_configurations(self, base, mesh_pos, verbose=False, tags=None):
        """Construct configurations"""
        configs = []
        base = _strip_labels(base.copy())

        for p in tqdm(mesh_pos, desc="creating configurations", disable=not verbose):
            framework = base.copy()
            framework.append(self.specie)
            framework.positions[-1] = p
            if tags:
                framework.info.update(tags)
            configs.append(framework)

        return configs


    def generate_dataset(
        self,
        defect_type: str,
        supercell_size: tuple | None = None,
        verbose: bool = False,
        dry_run: bool = False,
    ):
        """
        Generate defected structures

        Parameters
        ----------
        defect_type : str
            allowed types are "int", "vac", "frenkel"

        supercell_size : tuple | None, None by default
            repeat structure along each diagonal
            if None, self.r_cut will be used to estimate the size of the supercell
        
        verbose : bool, False by default
            verbosity

        dry_run : bool, False by default
            return only statistics for the dataset to be generated
        """

        if defect_type == 'int':
            sites_and_base = self._prepare_interstitials(supercell_size=supercell_size)
        elif defect_type == 'vac':
            sites_and_base = self._prepare_vacancies(supercell_size=supercell_size)
        elif defect_type == 'frenkel':
            sites_and_base = self._prepare_frenkels(supercell_size=supercell_size)
        else:
            raise ValueError(
                f"defect_type must be 'int', 'vac', or 'frenkel', got {defect_type!r}"
            )

        if dry_run:
            return self._statistics(sites_and_base)

        cfgs = []
        for site, (mesh_pos, base) in enumerate(sites_and_base):
            cfgs.extend(
                self._append_configurations(
                    base,
                    mesh_pos,
                    verbose=verbose,
                    tags={"defect": _DEFECT_NAMES[defect_type], "defect_site": site},
                )
            )
        return cfgs


def interstitials(
    atoms,
    specie,
    *,
    resolution: float = 0.25,
    r_cut: float = 6.0,
    r_min: float = 1.0,
    symprec: float | None = 0.1,
    supercell_size: tuple | None = None,
    dry_run: bool = False,
    verbose: bool = False,
):
    """
    Symmetrically distinct configurations with one extra mobile ion (interstitial).

    Parameters
    ----------
    atoms: ase.Atoms
        host structure (contains the mobile specie)
    specie: int
        atomic number of the mobile specie
    resolution, r_min, symprec:
        mesh spacing (A), blocking radius (A), symmetry precision, as in Grid
    r_cut: float, 6.0 by default
        the supercell is at least this thick in every direction (A)
    supercell_size: tuple, optional
        explicit repetitions; overrides r_cut
    dry_run: bool, False by default
        only return {'points': ..., 'bases': ...}

    Returns
    -------
    list of ase.Atoms with info['defect'] = 'interstitial'
    """
    sampler = _DefectSampler(atoms, specie, resolution=resolution, r_cut=r_cut, r_min=r_min,
                             symprec=symprec, verbose=verbose)
    return sampler.generate_dataset("int", supercell_size=supercell_size, verbose=verbose, dry_run=dry_run)


def vacancies(
    atoms,
    specie,
    *,
    resolution: float = 0.25,
    r_cut: float = 6.0,
    r_min: float = 1.0,
    symprec: float | None = 0.1,
    max_sep_dist: float | str = "auto",
    upper_bound: float = 6.0,
    supercell_size: tuple | None = None,
    dry_run: bool = False,
    verbose: bool = False,
):
    """
    Configurations for vacancy-mediated hops: for each distinct hop (edge of the migration
    network) both end sites are emptied and one mobile ion is placed on mesh points near the hop.

    Parameters
    ----------
    max_sep_dist: float or 'auto'
        keep points within this distance of the hop center; 'auto' = half the hop length + resolution
    upper_bound: float, 6.0 by default
        maximum hop length considered by the percolation analysis (A)

    Other parameters as in interstitials().

    Returns
    -------
    list of ase.Atoms with info['defect'] = 'vacancy' and info['defect_site'] = hop index
    """
    sampler = _DefectSampler(atoms, specie, resolution=resolution, r_cut=r_cut, r_min=r_min,
                             symprec=symprec, max_sep_dist=max_sep_dist, upper_bound=upper_bound,
                             verbose=verbose)
    return sampler.generate_dataset("vac", supercell_size=supercell_size, verbose=verbose, dry_run=dry_run)


def frenkel_pairs(
    atoms,
    specie,
    *,
    resolution: float = 0.25,
    r_cut: float = 6.0,
    r_min: float = 1.0,
    symprec: float | None = 0.1,
    max_sep_dist: float | str = "auto",
    upper_bound: float = 6.0,
    supercell_size: tuple | None = None,
    dry_run: bool = False,
    verbose: bool = False,
):
    """
    Frenkel pairs: for each symmetrically distinct mobile-ion site, the ion is moved to mesh
    points within max_sep_dist of its original site.

    Parameters
    ----------
    max_sep_dist: float or 'auto'
        maximum vacancy-interstitial separation (A); 'auto' = percolation cutoff of the network
    upper_bound: float, 6.0 by default
        maximum hop length for the percolation analysis (used only with max_sep_dist='auto')

    Other parameters as in interstitials().

    Returns
    -------
    list of ase.Atoms with info['defect'] = 'frenkel_pair' and info['defect_site'] = site index
    """
    sampler = _DefectSampler(atoms, specie, resolution=resolution, r_cut=r_cut, r_min=r_min,
                             symprec=symprec, max_sep_dist=max_sep_dist, upper_bound=upper_bound,
                             verbose=verbose)
    return sampler.generate_dataset("frenkel", supercell_size=supercell_size, verbose=verbose, dry_run=dry_run)
