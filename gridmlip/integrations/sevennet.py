#import torch
import numpy as np
from tqdm import tqdm

from torch_geometric.loader import DataLoader

#from sevenn.atom_graph_data import AtomGraphData
#from sevenn.train.graph_dataset import SevenNetGraphDataset
from sevenn.train import dataload
import sevenn.util as util
import sevenn._keys as KEY


def evaluate_atoms_list(
    atoms_list,
    model_path,
    batch_size=64,
    device="cuda",
    compute_force=False,
    allow_unlabeled=True,
    num_workers=1,
):
    """
    Predict energies for a list of atoms
    
    Parameters
    ----------

    atoms_list: list of ASE's atoms objects
        crystal structure configurations
        
    model_path: str
        path to the MACE model (e.g. "MACE-matpes-pbe-omat-ft.model")

    batch_size: int, 64 by default
        batch size

    device: str, 'cpu' by default
        device

    compute_force: boolean, False by default
        whether to predict forces

    allow_unlabeled: boolean, True by default
        allow unlabeled

    num_workers: int, 1 by default
        number of workers
        
    
    Returns
    ----------
    
    energies and forces (None if compute_force=False)
    """

    model, _ = util.model_from_checkpoint(model_path)
    model = model.to(device)
    # model = model.float() # check this line
    model.eval()
    model.set_is_batch_data(True)

    graph_list = dataload.graph_build(atoms_list,
                                      model.cutoff,
                                      allow_unlabeled=True,
                                      y_from_calc=True,
                                      num_cores=num_workers)


    loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False)

    energies_list = []
    forces_list = [] if compute_force else None

    for batch in tqdm(loader, desc = 'getting predictions'):
        batch = batch.to(device)
        out = model(batch).detach().cpu()
        out_list = util.to_atom_graph_list(out)

        for outg in out_list:
            outg = outg.fit_dimension()
            energies_list.append(float(outg[KEY.PRED_TOTAL_ENERGY]))
            if compute_force:
                forces_list.append(outg[KEY.PRED_FORCE])
    energies = np.array(energies_list)
    forces = np.concatenate(forces_list, axis=0) if compute_force else None
    return energies, forces
