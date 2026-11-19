# Changelog

## 0.3.0: new API

### Added
- `Grid.evaluate(model)` returns an `EnergyMap`. `model` can be an ASE calculator (MACE and SevenNet calculators are batched automatically), a predictor, any callable `list[Atoms] -> energies`, or a point model such as `BVSE`.
- `EnergyMap`: `percolation_barriers()`, `local_minima()`, `write(".grd"/".cube")`, `save()`/`load()` (`.npz`), `values`, `info` (model, settings, charges used).
- `Grid.points`, `len(grid)`, `Grid.prepare_configurations(supercell_cutoff=...)` (each configuration carries `info["grid_index"]`), `Grid.min_distances`, `Grid.summary()`.
- `Grid.load(energies | file)` returns an `EnergyMap`: a list/array of energies, or processed configurations (.cfg, .xyz, ...).
- `gridmlip.predictors.mace(model, head=, device=, dtype="float32", batch_size="auto")`: `model` is a path, a `mace_mp` name or a torch model. Multi-head models require `head=`.
- `gridmlip.predictors.sevennet(model, modal=...)`: not covered by tests yet.
- `gridmlip.predictors.gnn_lib(model, r_cut=5.0, n_max_neighbors=None)`: gnn_lib models such as CGCNN, batched; tested against a stand-in of the gnn_lib interface only.
- `gridmlip.BVSE(ion, ...)` and `gridmlip.bvse(atoms, ion)`. Oxidation states come from `atoms.arrays["oxi_states"]`, or are assigned with pymatgen's BVAnalyzer (bvlain settings plus `S-` forbidden, since there are no BVSE parameters for it), printed and recorded. Missing BVSE parameters raise a clear error naming the ions. If charges set inside `evaluate` break the grid symmetry, evaluation uses only the symmetry operations that keep the charges in place.
- `gridmlip.defects.interstitials / vacancies / frenkel_pairs`: configurations are tagged with `info["defect"]` and `info["defect_site"]`.
- Predictors: `pred.predict(atoms_list, properties)` returns any of `energy`, `forces`, `stress` (Voigt, eV/Å³, ASE convention) and per-atom `descriptors`, computed in the same batched pass for `mace`, `sevennet` and `gnn_lib`. MACE results are tested against `MACECalculator` (energy, forces, stress, `get_descriptors`). `pred(atoms_list)` still returns energies only.
- `EnergyMap.view(**kwargs)` / `gridmlip.view.view`: interactive 3D view, inline in Jupyter or as a standalone offline page (`.save()`, `.open()`). Shows atoms, bonds and coordination polyhedra by VESTA 3's default colors, pair rules and bond lengths, the mobile-ion sublattice, and an isosurface (default: 1D percolation barrier + 0.01 eV); hover labels, click-to-measure distances and angles, clickable polyhedra (CN, bond lengths, distortion index, volume), level slider, supercell up to 4×4×4, rotation free or about a/b/c/[uvw], light/dark theme. Bundles 3Dmol.js 2.5.5 (BSD-3-Clause, `gridmlip/view/3Dmol-LICENSE.txt`); needs only numpy, ase and scipy.
- `specie` accepts symbols ("Li") as well as atomic numbers.

- Dataset building blocks in `gridmlip.dataset`: `soap_features`, `select_fps` (PCA + FPS), `rattle` (reproducible perturbation keyed by seed, `material_id` and the configuration's content), `save_settings`/`load_settings` (settings + library versions).
- `gridmlip.dataset.build_dataset(structures, config, output_dir, ...)`: the function behind `gridmlip dataset`. Everything that defines the dataset, `specie` and `seed` included, comes only from the config; arguments only control the run (`output_dir`, `dry_run`, `overwrite`, `chunk`). Settings and structures are validated before any work: out-of-range values (all reported at once), duplicate or file-unsafe `material_id`s, structures without the mobile ion, and reuse of an output folder made with different settings (allowed with `overwrite`).
- `gridmlip dataset` command: YAML config (`--print-config`), resume by default, `--dry-run`, `--chunk i/n` for job arrays, `summary.csv`. Results do not depend on structure order or the chunk split.

### Fixed
- Generated configurations (defects, `prepare_configurations`) no longer inherit reference labels of the parent structure (`energy`, `REF_*`, forces, stress, ...). Before, they were written to `.xyz` and could be read back as if they were labels of the defect structures.
- `rattle_atoms`: if the input already has a pair closer than `d_min` (an ion placed at `r_min < d_min`), the check now uses that pair's distance. Before, every attempt failed and the structure was returned without position noise.

### Changed
- Installation: `pip install gridmlip` brings everything except dataset tools (numpy, scipy, ase, spglib, tqdm, ions, pymatgen). Extras: `fast` (numba, 10-100x faster percolation analysis; without it, importing gridmlip warns that the scipy backend is used) and `ml` (dscribe>=2, scikit-learn, fpsample, vesin, pyyaml for dataset generation). MLIP libraries (mace-torch, sevenn, gnn_lib) are integrations used when installed; the `mace`, `sevenn`, `gnn_lib`, `bvse`, `dscribe`, `scikit-learn` and `dataset` extras are removed. `requires-python = ">=3.10"`; the license classifier is corrected to MIT.
- `Grid` internals are private: `ir_grid_points`, `mapping`, `min_dists`, `mesh_frac`, `mesh_cart` and `generate_mesh()` now start with `_`. Use `grid.points`, `len(grid)`, `grid.min_distances` and `grid.mesh_shape` (now a property that builds the mesh on first access).
- Removed `gridmlip.select`, `gridmlip.perturb`, `gridmlip.utils` and `gridmlip.integrations.dscribe`; their code now lives in `gridmlip.dataset.selection` (`fps_with_stop`), `.rattle` (`rattle_atoms`, `rattle_positions`, `distort_cell_bounded`), `.settings` (`save_config`, `load_config`) and `.features` (`featurize_soap`).
- `Grid(verbose=False)` is now the default; call `grid.summary()` to see the mesh summary.
- `prepare_configurations(config_format="pymatgen")` now places the ion at Cartesian coordinates. Before, Cartesian values were passed as fractional ones.

### Removed
- `Grid.load_energies` (renamed to `Grid.load`), `Grid.construct_cfgs`, `read_processed_cfgs`, `percolation_barriers`, `write_grd`, `write_cube`, `min_distances_data`; `DefectGenerator`; the `gridmlip.integrations` package (`bvse.bvse_distribution`, `mace/sevennet.evaluate_atoms_list`, `gnn_lib.evaluate_atoms_list_cgcnn`). Replacements: `Grid.prepare_configurations`, `Grid.load`, `EnergyMap.percolation_barriers`, `EnergyMap.write`, `gridmlip.defects.*`, `gridmlip.bvse`, `gridmlip.predictors.*`.
- `gridmlip.dataset.generate_defects_dataset` (the 0.2.7 pipeline). Use `build_dataset` / `gridmlip dataset`, which run the same steps with a YAML config.

## 0.3.0: correctness fixes (included in 0.3.0)

### Fixed
- **Symmetry reduction could assign wrong values without warning.** Translations were rounded onto the mesh (`rint(t*(n-1))`), which is wrong for 1/3 and 1/6 translations (3₁, 6₁ screw axes, R-centering) and for 1/4 unless `n-1` happened to be divisible by 4. Cells that were symmetric only within `symprec` (e.g. a=4.00, b=4.04 Å) were also symmetrized using different numbers of points along the axes a rotation swaps. Worst errors in the min-distance field were 0.11–0.20 Å for quartz, P6₁, P4₁ and LiCoO₂, and 0.57 Å for the pseudo-tetragonal cell. Now each axis gets enough points for every operation to map the mesh onto itself, and any operation that still doesn't is dropped with a warning.
- Oxidation states (`atoms.arrays["oxi_states"]`) are now included in symmetry detection, so charge ordering (e.g. Fe²⁺/Fe³⁺) is no longer averaged out.
- The numba and scipy percolation backends now return identical, exact thresholds. Previously scipy bisected to 0.01 eV and returned the upper bound.
- Blocked points (`min_dist <= r_min`) and NaN predictions are stored as `+inf`. Previously they became 1.8e308. Volumetric files (`.grd`, `.cube`) replace them with the largest finite value.
- `Grid.load` checks that the number of energies matches the number of points.
- `percolation_radii()` and `min_distances_data` build the mesh if it doesn't exist yet. `percolation_barriers()` without loaded energies raises a clear error.
- MACE integration: progress bars were shown when `verbose=False` and hidden when `verbose=True`.
- Install hints now point to `gridmlip[fast]` and `gridmlip[dataset]`.

### Changed
- Installation: `pip install gridmlip` brings everything except dataset tools (numpy, scipy, ase, spglib, tqdm, ions, pymatgen). Extras: `fast` (numba, 10-100x faster percolation analysis; without it, importing gridmlip warns that the scipy backend is used) and `ml` (dscribe>=2, scikit-learn, fpsample, vesin, pyyaml for dataset generation). MLIP libraries (mace-torch, sevenn, gnn_lib) are integrations used when installed; the `mace`, `sevenn`, `gnn_lib`, `bvse`, `dscribe`, `scikit-learn` and `dataset` extras are removed. `requires-python = ">=3.10"`; the license classifier is corrected to MIT.
- **The mesh is now periodic:** point `i` sits at `i/n`, and the boundary plane is no longer repeated. `grid.data` has shape `n` (old: `n` including the duplicated plane), and `n` may be rounded up so the mesh fits the symmetry. Barriers differ from 0.2.7 within the usual mesh discretization error.
- Irreducible-point search uses O(N) memory: about 0.06 GB vs 5.3 GB for an 84³ Fd-3m mesh.
- `vesin`, `pyyaml`, `scikit-learn` and `dscribe` are imported lazily and listed in the new `dataset` extra. `joblib` was removed from the dependencies (unused).

### Tests
- New `tests/` suite (`pytest`; tests that run universal MLIPs are marked `umlip` and skipped by default, run them with `pytest -m umlip`): the symmetry-reduced field must equal the result without symmetry (Fm-3m, Pnma, R-3m, Fd-3m, P3₂21, P6₁, P4₁), the two backends must agree, plus grid I/O checks.
