from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from gnn_lib.graph_data import InMemoryDataset, atoms2data
from gnn_lib.data_utils import collate_fn


def evaluate_atoms_list_cgcnn(
    atoms_list,
    model_path,
    batch_size=64,
    r_cut=5.0,
    n_max_neighbors=None,
    device="cuda",
    compute_force=False,
    return_node_features=False,
):
    """
    Predict energies for a list of atoms
    
    Parameters
    ----------

    atoms_list: list of ASE's atoms objects
        crystal structure configurations
        
    model_path: str
        path to the MACE model (e.g. "cgcnn_matpes_pbe_small.pth")

    n_max_neighbors: int or None, None by default
        number of nearest neighbors
        use the same value used when training the model

    r_cut: float, 5.0 by default
        number of nearest neighbors
        use the same value used when training the model

    batch_size: int, 64 by default
        batch size

    device: str, 'cpu' by default
        device

    compute_force: boolean, False by default
        whether to predict forces
    
    Returns
    ----------
    
    energies and forces (None if compute_force=False)
    """

    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    
    for p in model.parameters():
        p.requires_grad = False

    
    data_list = [
        atoms2data(
            atoms,
            r_cut=r_cut,
            n_max_neighbors=n_max_neighbors,
        )
        for atoms in tqdm(atoms_list, desc='converting structures to graphs')
    ]

    dataset = InMemoryDataset.from_data_list(data_list)

    loader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                drop_last=False
            )
    
    energies_list = []
    forces_list = [] if compute_force else None
    node_feats_list = [] if return_node_features else None

    model.compute_forces=compute_force
    model.compute_stress=compute_force
    model.per_atom_target=False
    
    for batch in tqdm(loader, desc = 'getting predictions'):
        batch = batch.to(device)
        output = model(batch)
        energies_list.append(output['energy'].detach().cpu().numpy())
        if compute_force:
            forces_list.append(output['forces'].detach().cpu().numpy())
        if return_node_features:
            node_feats_list.append(output['node_feats'].detach().cpu().numpy())
            
        
        
    energies = np.concatenate(energies_list, axis=0).squeeze()
    forces = np.concatenate(forces_list, axis=0) if compute_force else None
    node_feats = np.concatenate(node_feats_list, axis=0) if return_node_features else None
    if return_node_features:
        return energies, forces, node_feats
    else:
        return energies, forces
