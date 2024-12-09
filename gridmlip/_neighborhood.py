import itertools 
import numpy as np
from scipy.spatial import cKDTree



def _scale_cell(cell, r_cut):
    volume = cell.volume
    scale_a = 2 * np.ceil(r_cut/(volume/np.linalg.norm(np.cross(cell[2, :], cell[1, :])))) + 1
    scale_b = 2 * np.ceil(r_cut/(volume/np.linalg.norm(np.cross(cell[0, :], cell[2, :])))) + 1
    scale_c = 2 * np.ceil(r_cut/(volume/np.linalg.norm(np.cross(cell[0, :], cell[1, :])))) + 1
    return scale_a, scale_b, scale_c



def nn_list(atoms_pos, mesh_pos, r_cut, cell = None):
    
    if np.any(cell):
        scale = _scale_cell(cell, r_cut)
        scales = []
        for s in scale:
            scales.append(np.arange(-(s // 2), s // 2 + 1))
        translations = tuple(list(itertools.product(
                                scales[0],
                                scales[1],
                                scales[2])))   
        
        buffer = (atoms_pos[:, None, :] + np.dot(translations, cell)).reshape(-1, 3, order = 'F')
    else:
        buffer = atoms_pos

    tree = cKDTree(buffer)
    distances, indexes = tree.query(mesh_pos, workers=-1, k=1, distance_upper_bound = r_cut)      
    return distances, indexes
