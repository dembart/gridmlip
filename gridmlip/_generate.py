import numpy as np
from tqdm import tqdm

from spglib import get_symmetry_dataset
from ions import Percolator, Edge

from .core import Grid
from ._neighborhood import nn_list
from ._distortion import rattle_positions, distort_lattice



class DefectGenerator:

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
        seed: int | np.random.Generator | None = None,
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
        self.rng = np.random.default_rng(seed)
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
        ir_mesh_cart = grid.mesh_cart[grid.ir_grid_points]
        filtered_pos = ir_mesh_cart[grid.min_dists >= grid.r_min]
        filtered_pos = self._filter_by_distance(filtered_pos, center, max_sep, atoms.cell)
        
        # filtering due to the lower defect concentration in a supercell
        defect_base = self._remove_atoms(base, remove_ids)
        
        filtered_pos = self._valid_positions(
            filtered_pos,
            defect_base[defect_base.numbers == self.specie],
            self.r_min,
        )
        
        return filtered_pos, defect_base


    def _prepare_interstitials(self):

        atoms = self.atoms.copy()
        grid = self._make_grid(atoms)
        base = grid._repeat_atoms(atoms, grid.r_cut)
        ir_mesh_cart = grid.mesh_cart[grid.ir_grid_points]
        filtered_pos = ir_mesh_cart[grid.min_dists >= grid.r_min]
        return [(filtered_pos, base)]



    def _prepare_vacancies(self):

        atoms = self.atoms.copy()
        pl, mincut, maxdim = self._get_percolator()
        edges = pl.unique_edges(cutoff=mincut, bottleneck_radius=0.0)

        base = Grid._repeat_atoms(atoms, self.r_cut)

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



    def _prepare_frenkels(self):

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

        base = Grid._repeat_atoms(atoms, self.r_cut)

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


    def _append_configurations(self, base, mesh_pos, pos_std=None, strain_std=None, verbose=False):
        """Construct configurations"""
        configs = []
    
        for p in tqdm(mesh_pos, desc="creating configurations", disable=not verbose):
            framework = base.copy()

            if pos_std is not None:
                framework = rattle_positions(framework, std=pos_std, rng=self.rng)  # rng added
    
            framework.append(self.specie)
            framework.positions[-1] = p
    
            if strain_std is not None:
                framework = distort_lattice(framework, std=strain_std, rng=self.rng)  # rng added
    
            configs.append(framework)
    
        return configs


    def generate_dataset(
        self,
        defect_type: str,
        pos_std: float | None = None,
        strain_std: float | None = None,
        verbose: bool = False,
        dry_run: bool = False,
    ):
        if defect_type == 'int':
            sites_and_base = self._prepare_interstitials()
        elif defect_type == 'vac':
            sites_and_base = self._prepare_vacancies()
        elif defect_type == 'frenkel':
            sites_and_base = self._prepare_frenkels()
        else:
            raise ValueError(
                f"defect_type must be 'int', 'vac', or 'frenkel', got {defect_type!r}"
            )

        if dry_run:
            return self._statistics(sites_and_base)

        cfgs = []
        for (mesh_pos, base) in sites_and_base:
            cfgs.extend(
                self._append_configurations(
                    base,
                    mesh_pos,
                    pos_std=pos_std,
                    strain_std=strain_std,
                    verbose=verbose,
                )
            )
        return cfgs