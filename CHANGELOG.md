# Changelog

## 0.3.0: new API

### Added
- `Grid.evaluate(model)` returns an `EnergyMap`. `model` can be an ASE calculator (MACE and SevenNet calculators are batched automatically), a predictor, any callable `list[Atoms] -> energies`, or a point model such as `BVSE`.
- `EnergyMap`: `percolation_barriers()`, `local_minima()`, `write(".grd"/".cube")`, `save()`/`load()` (`.npz`), `values`, `info` (model, settings, charges used).
- `Grid.points`, `len(grid)`, `Grid.prepare_configurations(supercell_cutoff=...)` (each configuration carries `info["grid_index"]`), `Grid.min_distances`, `Grid.summary()`.
- `Grid.load_energies(array | file)` returns an `EnergyMap`.
- `gridmlip.predictors.mace(model, head=, device=, dtype="float32", batch_size="auto")`: `model` is a path, a `mace_mp` name or a torch model. Multi-head models require `head=`.
- `gridmlip.predictors.sevennet(model, modal=...)`: not covered by tests yet.
- `gridmlip.BVSE(ion, ...)` and `gridmlip.bvse(atoms, ion)`. Oxidation states come from `atoms.arrays["oxi_states"]`, or are assigned with pymatgen's BVAnalyzer (bvlain settings), printed and recorded. If charges set inside `evaluate` break the grid symmetry, evaluation uses only the symmetry operations that keep the charges in place.
- `gridmlip.defects.interstitials / vacancies / frenkel_pairs`: configurations are tagged with `info["defect"]` and `info["defect_site"]`.
- `specie` accepts symbols ("Li") as well as atomic numbers.

- Dataset building blocks in `gridmlip.dataset`: `soap_features`, `select_fps` (PCA + FPS, same selection as 0.2.7), `rattle` (reproducible perturbation keyed by seed, `material_id` and the configuration's content; `key=j` reproduces 0.2.7), `save_settings`/`load_settings` (settings + library versions).
- `gridmlip dataset` command: YAML config (`--print-config`), resume by default, `--dry-run`, `--chunk i/n` for job arrays, `summary.csv`. Results do not depend on structure order or the chunk split.

### Fixed
- Generated configurations (defects, `prepare_configurations`) no longer inherit reference labels of the parent structure (`energy`, `REF_*`, forces, stress, ...). Before, they were written to `.xyz` and could be read back as if they were labels of the defect structures.
- `rattle_atoms`: if the input already has a pair closer than `d_min` (an ion placed at `r_min < d_min`), the check now uses that pair's distance. Before, every attempt failed and the structure was returned without position noise.

### Changed
- `Grid(verbose=False)` is now the default; call `grid.summary()` to see the mesh summary.
- `prepare_configurations(config_format="pymatgen")` now places the ion at Cartesian coordinates. Before, Cartesian values were passed as fractional ones.

### Deprecated (removed in 0.4)
- `Grid.construct_cfgs`, `read_processed_cfgs`, `percolation_barriers`, `write_grd`, `write_cube`; `DefectGenerator`; `integrations.bvse.bvse_distribution`; `integrations.mace/sevennet/gnn_lib.evaluate_atoms_list*`; `dataset.generate_defects_dataset`.

## 0.3.0: correctness fixes (included in 0.3.0)

### Fixed
- **Symmetry reduction could assign wrong values without warning.** Translations were rounded onto the mesh (`rint(t*(n-1))`), which is wrong for 1/3 and 1/6 translations (3₁, 6₁ screw axes, R-centering) and for 1/4 unless `n-1` happened to be divisible by 4. Cells that were symmetric only within `symprec` (e.g. a=4.00, b=4.04 Å) were also symmetrized using different numbers of points along the axes a rotation swaps. Worst errors in the min-distance field were 0.11–0.20 Å for quartz, P6₁, P4₁ and LiCoO₂, and 0.57 Å for the pseudo-tetragonal cell. Now each axis gets enough points for every operation to map the mesh onto itself, and any operation that still doesn't is dropped with a warning.
- Oxidation states (`atoms.arrays["oxi_states"]`) are now included in symmetry detection, so charge ordering (e.g. Fe²⁺/Fe³⁺) is no longer averaged out.
- The numba and scipy percolation backends now return identical, exact thresholds. Previously scipy bisected to 0.01 eV and returned the upper bound.
- Blocked points (`min_dist <= r_min`) and NaN predictions are stored as `+inf`. Previously they became 1.8e308. Volumetric files (`.grd`, `.cube`) replace them with the largest finite value.
- `load_energies` checks that the number of energies matches the number of points.
- `percolation_radii()` and `min_distances_data` build the mesh if it doesn't exist yet. `percolation_barriers()` without loaded energies raises a clear error.
- MACE integration: progress bars were shown when `verbose=False` and hidden when `verbose=True`.
- Install hints now point to `gridmlip[fast]` and `gridmlip[dataset]`.

### Changed
- **The mesh is now periodic:** point `i` sits at `i/n`, and the boundary plane is no longer repeated. `grid.data` has shape `n` (old: `n` including the duplicated plane), and `n` may be rounded up so the mesh fits the symmetry. Expect barriers to differ from 0.2.7 by the usual mesh discretization error (≤0.05 eV on a 0.2 Å mesh for the test set). Both versions converge to the same values on finer meshes.
- Irreducible-point search uses O(N) memory: about 0.06 GB vs 5.3 GB for an 84³ Fd-3m mesh.
- `vesin`, `pyyaml`, `scikit-learn` and `dscribe` are imported lazily and listed in the new `dataset` extra. `joblib` was removed from the dependencies (unused).

### Tests
- New `tests/` suite: the symmetry-reduced field must equal the result without symmetry (Fm-3m, Pnma, R-3m, Fd-3m, P3₂21, P6₁, P4₁), the two backends must agree, plus grid I/O checks. Regression against 0.2.7 uses `tests/data/reference_v0_2_7.json`, generated by `tests/make_reference.py` from the old code run without symmetry.
