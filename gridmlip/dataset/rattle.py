import hashlib
import warnings

import numpy as np
from ase import Atoms


def _get_rng(rng=None):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def distort_cell_bounded(
    atoms: Atoms,
    strain_std: float = 0.02,
    max_volume_change: float = 0.05,
    max_principal_strain: float = 0.05,
    max_abs_shear_strain: float = 0.05,
    rng: np.random.Generator | None = None,
    max_attempts: int = 100,
) -> Atoms:
    """
    Generate a randomly distorted cell with bounded volume change,
    bounded principal stretches, and bounded shear-strain components.
    """

    rng = _get_rng(rng)

    original_cell = atoms.get_cell().array.copy()

    lower_volume = 1.0 - max_volume_change
    upper_volume = 1.0 + max_volume_change

    for _ in range(max_attempts):
        eps = rng.normal(loc=0.0, scale=strain_std, size=(3, 3))
        eps = 0.5 * (eps + eps.T)

        eps_offdiag = eps - np.diag(np.diag(eps))
        if np.max(np.abs(eps_offdiag)) > max_abs_shear_strain:
            continue

        F = np.eye(3) + eps

        principal_stretches = np.linalg.svd(F, compute_uv=False)
        if (
            np.min(principal_stretches) < 1.0 - max_principal_strain
            or np.max(principal_stretches) > 1.0 + max_principal_strain
        ):
            continue

        volume_ratio = np.linalg.det(F)
        if not (lower_volume <= volume_ratio <= upper_volume):
            continue

        new_cell = original_cell @ F.T
        rattled = atoms.copy()
        # scale_atoms=True ensures atomic positions scale with the new cell boundaries
        rattled.set_cell(new_cell, scale_atoms=True)

        rattled.info["cell_rattle_strain_std"] = strain_std
        rattled.info["cell_rattle_strain"] = eps.tolist()
        rattled.info["cell_volume_ratio"] = volume_ratio
        rattled.info["cell_principal_stretches"] = principal_stretches.tolist()

        return rattled

    raise RuntimeError(
        "Could not generate a cell satisfying all deformation constraints."
    )


def rattle_positions(atoms, std=0.05, rng=None):
    """Randomly rattle atomic positions."""
    rng = _get_rng(rng)

    rattled = atoms.copy()
    rattled.rattle(stdev=std, rng=rng)
    rattled.arrays["original_positions"] = atoms.positions.copy()
    return rattled



def rattle_atoms(
    atoms: Atoms,
    pos_std: float = 0.05,
    strain_std: float = 0.02,
    max_volume_change: float = 0.05,
    max_principal_strain: float = 0.05,
    max_abs_shear_strain: float = 0.05,
    rng: np.random.Generator | None = None,
    max_attempts: int = 100,
    d_min: float = 1.4,
) -> Atoms:
    
    from vesin import ase_neighbor_list

    rattled = atoms.copy()
    valid_positions_found = False

    rng = _get_rng(rng)

    # If the unperturbed structure already has pairs closer than d_min (e.g. an ion placed at
    # r_min < d_min), require only that no pair gets closer than the original closest pair.
    _, _, d0 = ase_neighbor_list("ijd", atoms, cutoff=1.5 * d_min)
    threshold = min(d_min, float(np.min(d0))) if len(d0) else d_min

    for _ in range(max_attempts):
        rattled_candidate = rattle_positions(atoms, std=pos_std, rng=rng)

        i, j, d = ase_neighbor_list("ijd", rattled_candidate, cutoff=1.5 * d_min)

        if len(d) > 0 and np.min(d) < threshold:
            continue
            
        rattled = rattled_candidate
        valid_positions_found = True
        break
    
    if not valid_positions_found:
        warnings.warn(
            f"Could not satisfy minimum distance cutoff {threshold:.3f} Å within {max_attempts} attempts. "
            "Proceeding to cell distortion using the un-rattled structure.",
            UserWarning,
            stacklevel=2
        )

    try:
        rattled = distort_cell_bounded(
            rattled,
            strain_std=strain_std,
            max_volume_change=max_volume_change,
            max_principal_strain=max_principal_strain,
            max_abs_shear_strain=max_abs_shear_strain,
            rng=rng,
            max_attempts=max_attempts,
        )
    except RuntimeError as e:
        warnings.warn(
            f"Cell distortion failed: {e}. Returning position-rattled structure only.",
            UserWarning,
            stacklevel=2
        )

    return rattled



def _key_ints(data: bytes):
    digest = hashlib.blake2b(data, digest_size=16).digest()
    return tuple(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4))


def material_key(material_id):
    """4 integers from blake2b(str(material_id))."""
    return _key_ints(str(material_id).encode("utf-8"))


def content_key(atoms, decimals: int = 6):
    """4 integers identifying a structure by its atomic numbers, positions and cell."""
    data = b"".join([
        np.asarray(atoms.numbers, dtype=np.int64).tobytes(),
        np.round(np.asarray(atoms.positions, dtype=float), decimals).tobytes(),
        np.round(np.asarray(atoms.cell, dtype=float), decimals).tobytes(),
    ])
    return _key_ints(data)


def rattle_rng(atoms, seed=0, key=None, copy=0):
    """
    Random generator for rattling one configuration.

    The stream is SeedSequence(entropy=seed, spawn_key=(material key, configuration key[, copy])):
    - material key: from atoms.info['material_id'] (empty if absent)
    - configuration key: the structure's content (key=None), or an explicit int / tuple of ints;
      such as key=j, the index in the FPS selection

    Returns
    -------
    (np.random.Generator, spawn_key tuple)
    """
    mkey = material_key(atoms.info.get("material_id", ""))
    if key is None:
        ckey = content_key(atoms)
    elif isinstance(key, (int, np.integer)):
        ckey = (int(key),)
    else:
        ckey = tuple(int(k) for k in key)
    spawn_key = mkey + ckey + ((int(copy),) if copy else ())
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=spawn_key)), spawn_key


def rattle(
    atoms,
    seed: int = 0,
    *,
    key=None,
    copy: int = 0,
    rng: np.random.Generator | None = None,
    pos_std: float = 0.05,
    strain_std: float = 0.02,
    max_volume_change: float = 0.05,
    max_principal_strain: float = 0.05,
    max_abs_shear_strain: float = 0.05,
    d_min: float = 1.0,
    max_attempts: int = 100,
):
    """
    Reproducibly perturb one configuration: Gaussian noise on positions, then a bounded cell strain.

    The same structure, seed, key and copy always give the same result, independently of
    where the structure came from or in which order structures are processed.

    Parameters
    ----------
    atoms: ase.Atoms
    seed: int, 0 by default
        dataset-wide seed
    key: None, int or tuple of ints
        identifies the configuration; None = its content (see rattle_rng)
    copy: int, 0 by default
        index of the perturbed copy, for several independent copies of one configuration
    rng: np.random.Generator, optional
        use this generator instead (seed, key and copy are then ignored)
    pos_std: float, 0.05 by default
        standard deviation of position noise (A); 0 disables it
    strain_std: float, 0.02 by default
        standard deviation of strain components; 0 disables the cell distortion
    max_volume_change, max_principal_strain, max_abs_shear_strain:
        bounds of the cell distortion
    d_min: float, 1.0 by default
        no pair may get closer than d_min (or than the closest pair of the input, if that is closer)

    Returns
    -------
    ase.Atoms with info['rattle_seed'], info['rattle_key'], info['rattle_copy']
    """
    if rng is None:
        rng, spawn_key = rattle_rng(atoms, seed=seed, key=key, copy=copy)
        record = {"rattle_seed": int(seed), "rattle_key": " ".join(map(str, spawn_key)), "rattle_copy": int(copy)}
    else:
        record = {"rattle_seed": "external rng"}

    out = rattle_atoms(atoms, pos_std=pos_std, strain_std=strain_std, max_volume_change=max_volume_change,
                       max_principal_strain=max_principal_strain, max_abs_shear_strain=max_abs_shear_strain,
                       rng=rng, max_attempts=max_attempts, d_min=d_min)
    out.info.update(record)
    return out
