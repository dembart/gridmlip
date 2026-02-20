import torch
import numpy as np
from tqdm import tqdm

from mace import data
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils


def evaluate_atoms_list(
    atoms_list,
    model_path,
    batch_size=64,
    device="cuda",
    compute_force=False,
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
    
    Returns
    ----------
    
    energies and forces (None if compute_force=False)
    """


    device = torch_tools.init_device(device)

    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.float() # check this line
    model = model.to(device)
    
    for p in model.parameters():
        p.requires_grad = False

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
    
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    dataset = [
        data.AtomicData.from_config(
            cfg,
            z_table=z_table,
            cutoff=float(model.r_max),
            heads=heads,
        )
        for cfg in tqdm(configs, desc='converting structures to graphs')
    ]

    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    energies_list = []
    forces_list = [] if compute_force else None

    for batch in tqdm(loader, desc = 'getting predictions'):
        batch = batch.to(device)
        output = model(batch.to_dict(), compute_force=compute_force)
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if compute_force:
            forces_list.append(torch_tools.to_numpy(output["forces"]))
        
    energies = np.concatenate(energies_list, axis=0)
    forces = np.concatenate(forces_list, axis=0) if compute_force else None
    return energies, forces