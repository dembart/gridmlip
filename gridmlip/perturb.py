import warnings
import numpy as np
from ase import Atoms
from vesin import ase_neighbor_list


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
    
    rattled = atoms.copy()
    valid_positions_found = False

    rng = _get_rng(rng)

    for _ in range(max_attempts):
        rattled_candidate = rattle_positions(atoms, std=pos_std, rng=rng)
        
        i, j, d = ase_neighbor_list("ijd", rattled_candidate, cutoff=1.5 * d_min)
        
        if len(d) > 0 and np.min(d) < d_min:
            continue
            
        rattled = rattled_candidate
        valid_positions_found = True
        break
    
    if not valid_positions_found:
        warnings.warn(
            f"Could not satisfy minimum distance cutoff {d_min} Å within {max_attempts} attempts. "
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
