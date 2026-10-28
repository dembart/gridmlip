import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from ase.cell import Cell
 
 
def _find_local_minima(distribution, size=3):
    E = distribution[:-1, :-1, :-1]
    minf = ndimage.minimum_filter(E, size=size, mode='wrap')
    mask = E == minf
 
    padded = np.pad(mask, 1, mode='wrap')
    labeled, _ = ndimage.label(padded, structure=np.ones((3, 3, 3), bool))
    labeled = labeled[1:-1, 1:-1, 1:-1]
 
    labels = np.unique(labeled)
    labels = labels[labels != 0]
    idx = np.array([np.argwhere(labeled == l)[0] for l in labels])
    val = E[tuple(idx.T)]
    return idx, val
 
 
def _cluster_minima(idx, val, shape, cell, min_distance, max_value=None):
    if max_value is not None:
        m = val < max_value
        idx, val = idx[m], val[m]
 
    cell = np.asarray(cell, dtype=float)
    frac = idx / np.array(shape, dtype=float)
    cart = frac @ cell
 
    shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')).reshape(3, -1).T
    n = len(idx)
    rep_cart = ((shifts[:, None, :] + frac[None, :, :]).reshape(-1, 3)) @ cell
    owner = np.tile(np.arange(n), len(shifts))
    tree = cKDTree(rep_cart)
 
    order = np.argsort(val)
    alive = np.ones(n, bool)
 
    clusters = []
    for i in order:
        if not alive[i]:
            continue
        neighbors = np.unique(owner[tree.query_ball_point(cart[i], min_distance)])
        clusters.append((idx[i], val[i], len(neighbors)))
        for j in neighbors:
            alive[j] = False
 
    return clusters


def clustered_local_minima(
        data: np.ndarray, 
        min_distance: float, 
        relative_max_value: float, 
        cell: np.ndarray,
        coord_type : str = 'cart',
    ):
    """
    Find local minima for calculated data on a grid.

    Parameters
    ----------
    data : np.ndarray of shape (N, M, K)
        data calculated for the generated grid

    min_distance : float
        minimal distance between voids

    relative_max_value : float
        relative maximum value of the data to search for minima
        `data.min() + relative_max_value`
    
    cell : np.ndarray of size (3,3)
        lattice matrix

    coord_type : str, "cart" by default
        in what format minima coordinates will be return
        allowed values are "cart", "frac", mesh"
        "cart" - cartesian coordinates
        "frac" - fractional coordinates
        "mesh" - mesh posititions

    Returns
    -------
    clusters : list of lists, 
        each list contains position of a minimum,
        energy value, and number of points clustered

    """
    assert coord_type in ("cart", "frac", "mesh")

    idx, val = _find_local_minima(data)
    core_shape = tuple(s - 1 for s in data.shape)
    clusters = _cluster_minima(idx, val, core_shape,
                              min_distance=min_distance,
                              max_value=data.min() + relative_max_value,
                              cell=cell,
                             )

    if coord_type == "mesh":
        return clusters
    else: 
        cell_ase = Cell(cell)
        clusters_new = []
        for i, v, n in clusters:
            scaled_site = i/(np.array(data.shape) - 1)
            if coord_type == "cart":
                cart_site = cell_ase.cartesian_positions(scaled_site)
                clusters_new.append([cart_site, v, n])
            else:
                clusters_new.append([scaled_site, v, n])
        return clusters_new