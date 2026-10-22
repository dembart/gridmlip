import numpy as np


def _get_rng(rng=None):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def distort_lattice(atoms, std=0.03, rng=None):
    """Randomly apply independent strain to the three lattice vectors."""
    rng = _get_rng(rng)

    cell = atoms.cell.array.copy()
    lengths = np.linalg.norm(cell, axis=1)

    scale = rng.uniform(
        1 - std,
        1 + std,
        size=3,
    )

    new_lengths = lengths * scale
    new_cell = cell * (new_lengths / lengths)[:, None]

    distorted = atoms.copy()
    distorted.info["original_cell"] = atoms.cell.array.copy()
    distorted.set_cell(new_cell, scale_atoms=True)

    return distorted


def rattle_positions(atoms, std=0.05, rng=None):
    """Randomly rattle atomic positions."""
    rng = _get_rng(rng)

    rattled = atoms.copy()
    rattled.arrays["original_positions"] = (
        rattled.positions.copy()
    )

    rattled.rattle(stdev=std, rng=rng)

    return rattled