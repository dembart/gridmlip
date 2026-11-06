import numpy as np
from tqdm import tqdm

def evaluate_atoms_list(
    atoms_list,
    model_path,
    batch_size=64,
    dtype="float32",
    device="cuda",
    compute_force=False,
    compute_stress=False,
    verbose=True,
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
        predict forces

    compute_stress: boolean, False by default
        predict stress

    dtype: str,
        "float32" or "float64"
    
    Returns
    ----------
    
    energies and forces (None if compute_force=False)
    """

    try:
        import torch
        from mace import data
        from mace.tools import torch_geometric, torch_tools, utils
    except ImportError as exc:
        raise ImportError(
            "mace library is required."
            "Install with `pip install 'gridmlip[mace]'`."
        ) from exc

    assert dtype in ("float32", "float64")
    
    device = torch_tools.init_device(device)

    model = torch.load(model_path, map_location=device, weights_only=False)
    
    model = model.float() if dtype=="float32" else model.double() # check this line
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
        for cfg in tqdm(configs, desc='converting structures to graphs', disable=verbose)
    ]

    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    energies_list = []
    forces_list = [] if compute_force else None
    stress_list = [] if compute_stress else None

    target_dtype = torch.float32 if dtype=='float32' else torch.float64
    
    for batch in tqdm(loader, desc = 'getting predictions', disable=verbose):
        batch = batch.to(device)
        for key in batch.keys:
            value = getattr(batch, key)
            if torch.is_tensor(value) and torch.is_floating_point(value):
                setattr(batch, key, value.to(target_dtype))
        output = model(batch.to_dict(), compute_force=compute_force, compute_stress=compute_stress)
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if compute_force:
            forces_list.append(torch_tools.to_numpy(output["forces"]))
        if compute_stress:
            stress_list.append(torch_tools.to_numpy(output["stress"]))
        
    energies = np.concatenate(energies_list, axis=0)
    
    forces = np.concatenate(forces_list, axis=0) if compute_force else None
    n_nodes = list(map(len, atoms_list))
    forces = np.split(forces, np.cumsum(n_nodes)[:-1]) if compute_force else None
    
    stress = np.concatenate(stress_list, axis=0) if compute_stress else None

    return energies, forces, stress