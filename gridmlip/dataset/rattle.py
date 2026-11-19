import hashlib

import numpy as np

from ..perturb import rattle_atoms


def _key_ints(data: bytes):
    digest = hashlib.blake2b(data, digest_size=16).digest()
    return tuple(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4))


def material_key(material_id):
    """4 integers from blake2b(str(material_id)); same as gridmlip <= 0.2.7."""
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
      key=j with copy=0 reproduces gridmlip <= 0.2.7 (j = index in the FPS selection)

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
