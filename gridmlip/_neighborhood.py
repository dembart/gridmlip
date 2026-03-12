import itertools 
import numpy as np
from scipy.spatial import cKDTree


def nn_list(atoms_pos, mesh_pos, r_cut, cell=None):

    if np.any(cell):
        volume = np.linalg.det(cell)
        h1 = volume/np.linalg.norm(np.cross(cell[2, :], cell[1, :]))
        h2 = volume/np.linalg.norm(np.cross(cell[0, :], cell[2, :]))
        h3 = volume/np.linalg.norm(np.cross(cell[0, :], cell[1, :]))
        scale = 2 * np.array(np.ceil(r_cut/np.array([h1, h2, h3])), dtype=int) + 1
        scales = []
        for s in scale:
            scales.append(np.arange(-(s // 2), s // 2 + 1))

        translations = tuple(list(itertools.product(
                                scales[0],
                                scales[1],
                                scales[2])))   
        
        buffer = (atoms_pos[:, None, :] + np.dot(translations, cell)).reshape(-1, 3, order='F')
    else:
        buffer = atoms_pos

    tree = cKDTree(buffer)
    distances, indexes = tree.query(mesh_pos, workers=-1, k=1, distance_upper_bound=r_cut)      
    return distances, indexes
