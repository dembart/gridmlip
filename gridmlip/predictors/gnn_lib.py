from pathlib import Path

import numpy as np
from tqdm import tqdm

from .base import Predictor, resolve_device, split_per_atom, to_voigt


def _import_gnn_lib():
    try:
        import torch
        from torch.utils.data import DataLoader
        from gnn_lib.graph_data import InMemoryDataset, atoms2data
        from gnn_lib.data_utils import collate_fn
    except ImportError as exc:
        raise ImportError("gnn_lib is required. Install it with `pip install gnn_lib`.") from exc
    return torch, DataLoader, InMemoryDataset, atoms2data, collate_fn


class GNNLibPredictor(Predictor):
    """
    Batched predictions of a gnn_lib model (e.g. CGCNN) for a list of structures.

    ``predictor(atoms_list)`` -> energies; ``predictor.predict(atoms_list, properties)`` -> energy,
    forces, stress and/or descriptors (the model's 'node_feats'), from the same batched pass.
    Use ``gridmlip.predictors.gnn_lib(...)`` to create one.

    Note: written against the gnn_lib API used by gridmlip 0.2.7
    (``atoms2data``, ``InMemoryDataset``, ``collate_fn``); tested only with a stand-in for it.
    """

    supported = ("energy", "forces", "stress", "descriptors")

    def __init__(self, model, *, r_cut=5.0, n_max_neighbors=None, device="cpu", batch_size=64,
                 progress=True, source=None):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # total (not per-atom) energies; forces/stress are switched on per call
        self.model.compute_forces = False
        self.model.compute_stress = False
        self.model.per_atom_target = False
        self.r_cut = r_cut
        self.n_max_neighbors = n_max_neighbors
        self.batch_size = int(batch_size)
        self.progress = progress
        self.info = {"model": f"gnn_lib ({source})" if source else "gnn_lib", "source": source,
                     "r_cut": r_cut, "n_max_neighbors": n_max_neighbors, "device": str(device),
                     "batched": True}

    def _predict(self, atoms_list, properties):
        torch, DataLoader, InMemoryDataset, atoms2data, collate_fn = _import_gnn_lib()
        need_grad = "forces" in properties or "stress" in properties
        self.model.compute_forces = "forces" in properties
        self.model.compute_stress = "stress" in properties

        res = {p: [] for p in properties}
        with tqdm(total=len(atoms_list), desc="getting predictions", disable=not self.progress) as bar:
            # graphs are built per chunk, so memory does not grow with the grid size
            for start in range(0, len(atoms_list), self.batch_size):
                chunk = atoms_list[start:start + self.batch_size]
                data = [atoms2data(a, r_cut=self.r_cut, n_max_neighbors=self.n_max_neighbors) for a in chunk]
                loader = DataLoader(InMemoryDataset.from_data_list(data), batch_size=len(chunk),
                                    collate_fn=collate_fn, shuffle=False, drop_last=False)
                batch = next(iter(loader)).to(self.device)
                with torch.set_grad_enabled(need_grad):
                    out = self.model(batch)

                def np_(key):
                    if key not in out or out[key] is None:
                        raise NotImplementedError(f"this gnn_lib model does not return {key!r}")
                    return out[key].detach().cpu().double().numpy()

                if "energy" in res:
                    e = np_("energy").reshape(-1)
                    if e.size != len(chunk):
                        raise RuntimeError(f"gnn_lib model returned {e.size} energies for {len(chunk)} "
                                           "structures (is per_atom_target supported by this model?)")
                    res["energy"].append(e)
                if "forces" in res:
                    res["forces"] += split_per_atom(np_("forces"), chunk)
                if "stress" in res:
                    st = np_("stress")  # (n, 3, 3) or (n, 6); sign convention as returned by the model
                    st = st.reshape(len(chunk), 3, 3) if st.size == 9 * len(chunk) else st.reshape(len(chunk), 6)
                    res["stress"].append(to_voigt(st))
                if "descriptors" in res:
                    res["descriptors"] += split_per_atom(np_("node_feats"), chunk)
                bar.update(len(chunk))

        for p in ("energy", "stress"):
            if p in res:
                res[p] = np.concatenate(res[p])
        return res


def gnn_lib(model, *, r_cut=5.0, n_max_neighbors=None, device="auto", batch_size=64, progress=True):
    """
    Batched gnn_lib predictor (e.g. CGCNN).

    Parameters
    ----------
    model: str, path or torch model
        path to a model saved with torch.save, or the model itself

    r_cut: float, 5.0 by default
        graph cutoff; use the value the model was trained with

    n_max_neighbors: int or None
        maximum number of neighbors; use the value the model was trained with

    device: str, 'auto' by default
        'auto' (cuda if available, else cpu), 'cuda', 'cpu'

    batch_size: int, 64 by default

    progress: bool, True by default

    Returns
    -------
    GNNLibPredictor: ``pred(atoms_list)`` -> energies;
    ``pred.predict(atoms_list, ("energy", "forces", "stress", "descriptors"))`` -> dict
    """
    torch, *_ = _import_gnn_lib()
    dev = resolve_device(device)
    if isinstance(model, torch.nn.Module):
        module, source = model, type(model).__name__
    else:
        path = Path(str(model)).expanduser()
        if not path.is_file():
            raise ValueError(f"{model!r} is not an existing model file")
        module, source = torch.load(path, map_location=dev, weights_only=False), str(path)
    return GNNLibPredictor(module, r_cut=r_cut, n_max_neighbors=n_max_neighbors, device=dev,
                           batch_size=batch_size, progress=progress, source=source)
