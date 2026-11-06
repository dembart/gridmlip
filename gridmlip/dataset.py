
import hashlib
from pathlib import Path

import numpy as np

from ase.io import write
from sklearn.decomposition import PCA

from .core import Grid
from .defects import DefectGenerator
from .utils import save_config
from .select import fps_with_stop
from .perturb import rattle_atoms
from .integrations.dscribe import featurize_soap



def _make_rng(master_seed, material_id, config_id):
    digest = hashlib.blake2b(
        str(material_id).encode("utf-8"),
        digest_size=16,
    ).digest()

    material_key = tuple(
        int.from_bytes(digest[i:i + 4], "little")
        for i in range(0, 16, 4)
    )

    seed = np.random.SeedSequence(
        entropy=master_seed,
        spawn_key=material_key + (int(config_id),),
    )

    return np.random.default_rng(seed)


def generate_defects_dataset(
    atoms_list,
    *,
    output_dir="./dataset",
    overwrite=True,

    # Defect generation
    specie=None,
    resolution=0.25,
    symprec=0.1,
    r_min=1.5,
    r_cut=5.0,
    max_sep_dist="auto",
    upper_bound=6.0,
    defect_type="all",
    supercell_size=None,
    verbose=False,
    dry_run=False,
    max_configs=500000,
    max_atoms_in_cfg=150,

    # SOAP
    soap_r_cut=6.0,
    soap_n_max=3,
    soap_l_max=2,
    soap_sigma=0.3,
    soap_periodic=True,
    soap_dtype="float64",
    soap_normalize=True,

    # PCA and FPS
    pca_n_components=0.95,
    max_samples=500,
    rel_tol=1e-5,
    start_idx="centroid",

    # Augmentation
    master_seed=42,
    d_min=1.0,
    pos_std=0.05,
    strain_std=0.02,
    max_volume_change=0.05,
    max_principal_strain=0.05,
    max_abs_shear_strain=0.05,
    max_attempts=100,
):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all function arguments except atoms_list
    config = {
        k: v for k, v in locals().items()
        if k != "atoms_list"
    }

    save_config(config, output_dir)

    if defect_type == "all":
        defect_types = ("int", "vac", "frenkel")
    else:
        defect_types = (defect_type,)

    generated_configs = []

    for i, atoms in enumerate(atoms_list):

        material_id = atoms.info.get("material_id", i)
        output_file = output_dir / f"dataset_{material_id}.xyz"

        if output_file.exists() and not overwrite:
            print(f"Skipping existing file: {output_file}")
            continue

        base = Grid._repeat_atoms(atoms, r_cut=r_cut, supercell_size=supercell_size)
        if len(base) + 1 > max_atoms_in_cfg:
            print(f"Skipping current structure (not passing max_atoms_in_cfg ({len(base)} > {max_atoms_in_cfg}))")
            continue

        gen = DefectGenerator(
            atoms,
            specie=specie,
            resolution=resolution,
            symprec=symprec,
            r_min=r_min,
            r_cut=r_cut,
            max_sep_dist=max_sep_dist,
            upper_bound=upper_bound,
        )
        
        n_cfgs = 0
        for defect in defect_types:
            n_cfgs += gen.generate_dataset(
                    defect_type=defect,
                    dry_run=True,
                    supercell_size=supercell_size,
                    verbose=verbose,
                )['points']

        if dry_run:
            generated_configs.append(n_cfgs)
            continue
            
        if n_cfgs > max_configs:
            print(f"Skipping current structure (not passing max_configs ({n_cfgs} > {max_configs}))")
            continue

        cfgs = []
        for defect in defect_types:
            cfgs.extend(
                gen.generate_dataset(
                    defect_type=defect,
                    dry_run=dry_run,
                    supercell_size=supercell_size,
                    verbose=verbose,
                )
            )

        if not cfgs:
            continue

        print(f"material_id: {material_id} \n\tgenerated {len(cfgs)} configurations")

        features = featurize_soap(
            cfgs,
            site_id=-1,
            r_cut=soap_r_cut,
            n_max=soap_n_max,
            l_max=soap_l_max,
            sigma=soap_sigma,
            periodic=soap_periodic,
            dtype=soap_dtype,
            normalize=soap_normalize,
        )

        # Explicit full SVD avoids randomized PCA solvers.
        X = PCA(
            n_components=pca_n_components,
            svd_solver="full",
        ).fit_transform(features)

        sampled_ids = fps_with_stop(
            X,
            rel_tol=rel_tol,
            start_idx=start_idx,
            max_samples=min(max_samples, len(X)),
        )

        print(f"material_id: {material_id} \n\tsampled {len(sampled_ids)} configurations")

        sampled_configs = []

        for j in sampled_ids:
            rng = _make_rng(
                master_seed=master_seed,
                material_id=material_id,
                config_id=j,
            )

            rattled = rattle_atoms(
                cfgs[j],
                rng=rng,
                d_min=d_min,
                pos_std=pos_std,
                strain_std=strain_std,
                max_volume_change=max_volume_change,
                max_principal_strain=max_principal_strain,
                max_abs_shear_strain=max_abs_shear_strain,
                max_attempts=max_attempts,
            )

            sampled_configs.append(rattled)

        write(output_file, sampled_configs)
        generated_configs.append(sampled_configs)

    return generated_configs

