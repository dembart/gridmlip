import itertools
import numpy as np
from scipy.spatial import cKDTree


def nn_list(atoms_pos, mesh_pos, r_cut, cell=None, k=1, return_orig_ids=False):
    """
    k nearest neighbors for each query point in `mesh_pos`.

    Parameters
    ----------
    atoms_pos : (N, 3) ndarray
        Atom positions.

    mesh_pos : (M, 3) ndarray
        Query points.

    r_cut : float
        Cutoff

    cell : (3, 3) ndarray or None
        Periodic cell. If given, periodic images within `r_cut` are added.

    k : int
        Number of neighbors.

    Returns
    -------
    distances : (M, k) ndarray
        Distances. `inf` where no neighbor was found within `r_cut`.
    
    indexes : (M, k) ndarray
        Indices into the buffered array
    
    orig_indexes : (M, k) ndarray
        Indices into the original `atoms_pos`
    """
    atoms_pos = np.asarray(atoms_pos)
    n_atoms = len(atoms_pos)

    if cell is not None and np.any(cell):
        cell = np.asarray(cell)
        volume = np.linalg.det(cell)
        h1 = volume / np.linalg.norm(np.cross(cell[2, :], cell[1, :]))
        h2 = volume / np.linalg.norm(np.cross(cell[0, :], cell[2, :]))
        h3 = volume / np.linalg.norm(np.cross(cell[0, :], cell[1, :]))
        scale = 2 * np.array(np.ceil(r_cut / np.array([h1, h2, h3])), dtype=int) + 1
        scales = [np.arange(-(s // 2), s // 2 + 1) for s in scale]

        translations = np.array(list(itertools.product(*scales)))  # (T, 3)
        n_trans = len(translations)

        # (n_atoms, n_trans, 3) -> (n_atoms * n_trans, 3), C-order:
        # buffer row k = i * n_trans + t
        buffer = (atoms_pos[:, None, :] + translations @ cell).reshape(-1, 3)

        # Consistent with the C-order reshape above.
        buffer_to_orig = np.repeat(np.arange(n_atoms), n_trans)
    else:
        buffer = atoms_pos
        buffer_to_orig = np.arange(n_atoms)

    tree = cKDTree(buffer)
    distances, indexes = tree.query(
        mesh_pos, workers=-1, k=k, distance_upper_bound=r_cut
    )

    if not return_orig_ids:
        return distances, indexes

    # cKDTree returns len(buffer) for "not found". Map valid entries back
    # to original atom indices; leave "not found" as -1.
    orig_indexes = np.full_like(indexes, -1)
    valid = indexes < len(buffer)
    orig_indexes[valid] = buffer_to_orig[indexes[valid]]

    return distances, indexes, orig_indexes