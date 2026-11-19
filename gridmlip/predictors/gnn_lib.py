import numpy as np
from tqdm import tqdm

from .base import Predictor, resolve_device, split_per_atom, to_voigt

_INSTALL = "gnn_lib is required: pip install git+https://github.com/dembart/gnn-lib"


def _import_gnn_lib():
    try:
        import torch
        from gnn_lib.graph_data import atoms2data
        from gnn_lib.data_utils import collate_fn
    except ImportError as exc:
        raise ImportError(_INSTALL) from exc
    return torch, atoms2data, collate_fn


class GNNLibPredictor(Predictor):
    """
    Batched predictions of a gnn_lib model (CGCNN) for a list of structures.

    ``predictor(atoms_list)`` -> energies; ``predictor.predict(atoms_list, properties)`` -> energy,
    forces and/or stress, from the same batched pass. Energies are total energies computed exactly
    as gnn_lib's CGCNNCalculator does (per-atom outputs times the number of atoms, plus the model's
    per-element reference energies). Use ``gridmlip.predictors.gnn_lib(...)`` to create one.
    """

    supported = ("energy", "forces", "stress")

    def __init__(self, model, *, r_cut, n_max_neighbors=None, device="cpu", batch_size=64,
                 progress=True, source=None):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.r_cut = float(r_cut)
        self.n_max_neighbors = n_max_neighbors
        self.batch_size = int(batch_size)
        self.progress = progress
        self.info = {"model": f"gnn_lib ({source})" if source else "gnn_lib", "source": source,
                     "r_cut": self.r_cut, "n_max_neighbors": n_max_neighbors, "device": str(device),
                     "batched": True}

    @classmethod
    def from_calculator(cls, calc, batch_size=64, progress=True):
        """Batched predictor from a gnn_lib CGCNNCalculator (same model, cutoffs and device)."""
        data_cfg = calc.config.data
        return cls(calc.model, r_cut=data_cfg.r_cut, n_max_neighbors=data_cfg.n_max_neighbors or None,
                   device=calc.device, batch_size=batch_size, progress=progress,
                   source=type(calc).__name__)

    def _total_energy(self, energy, batch):
        """Same as CGCNNCalculator: add reference energies, then per-atom -> total."""
        if getattr(self.model, "ref_energies", None) is not None:
            energy = self.model._energy_from_e0s_and_residuals(energy, batch)
        energy = energy.reshape(-1)
        if getattr(self.model, "per_atom_target", False):
            energy = energy * batch.num_atoms.to(energy.dtype)
        return energy

    def _predict(self, atoms_list, properties):
        torch, atoms2data, collate_fn = _import_gnn_lib()
        need_grad = "forces" in properties or "stress" in properties
        saved = (getattr(self.model, "compute_forces", False), getattr(self.model, "compute_stress", False))
        self.model.compute_forces = "forces" in properties
        self.model.compute_stress = "stress" in properties

        res = {p: [] for p in properties}
        try:
            with tqdm(total=len(atoms_list), desc="getting predictions", disable=not self.progress) as bar:
                # graphs are built per chunk, so memory does not grow with the grid size
                for start in range(0, len(atoms_list), self.batch_size):
                    chunk = atoms_list[start:start + self.batch_size]
                    batch = collate_fn([atoms2data(a, r_cut=self.r_cut, n_max_neighbors=self.n_max_neighbors)
                                        for a in chunk]).to(self.device)
                    with torch.set_grad_enabled(need_grad):
                        out = self.model(batch)

                    if "energy" in res:
                        e = self._total_energy(out["energy"], batch).detach().cpu().double().numpy()
                        if e.size != len(chunk):
                            raise RuntimeError(f"gnn_lib model returned {e.size} energies for {len(chunk)} structures")
                        res["energy"].append(e)
                    if "forces" in res:
                        if out.get("forces") is None:
                            raise NotImplementedError("this gnn_lib model does not return forces")
                        res["forces"] += split_per_atom(out["forces"].detach().cpu().double().numpy(), chunk)
                    if "stress" in res:
                        if out.get("stress") is None:
                            raise NotImplementedError("this gnn_lib model does not return stress")
                        st = out["stress"].detach().cpu().double().numpy()  # (n, 3, 3), as in CGCNNCalculator
                        st = st.reshape(len(chunk), 3, 3) if st.size == 9 * len(chunk) else st.reshape(len(chunk), 6)
                        res["stress"].append(to_voigt(st))
                    bar.update(len(chunk))
        finally:
            self.model.compute_forces, self.model.compute_stress = saved

        for p in ("energy", "stress"):
            if p in res:
                res[p] = np.concatenate(res[p])
        return res


def gnn_lib(model="matpes_pbe_small", *, r_cut=None, n_max_neighbors=None, device="auto", batch_size=64,
            progress=True):
    """
    Batched gnn_lib predictor (CGCNN).

    Parameters
    ----------
    model: str, path or torch model, "matpes_pbe_small" by default
        a pretrained name known to gnn_lib ("matpes_pbe_small", downloaded and cached by gnn_lib),
        a checkpoint path, or an initialized model (then r_cut is required)

    r_cut, n_max_neighbors: optional
        graph settings; by default taken from the model's config (use the training values)

    device: str, 'auto' by default
        'auto' (cuda if available, else cpu), 'cuda', 'cpu'

    batch_size: int, 64 by default

    progress: bool, True by default

    Returns
    -------
    GNNLibPredictor: ``pred(atoms_list)`` -> energies;
    ``pred.predict(atoms_list, ("energy", "forces", "stress"))`` -> dict
    """
    torch, *_ = _import_gnn_lib()
    dev = resolve_device(device)
    if isinstance(model, torch.nn.Module):
        cfg = getattr(model, "config", None)
        if r_cut is None:
            if cfg is None:
                raise ValueError("r_cut is required for a model without a config")
            r_cut = cfg.data.r_cut
        if n_max_neighbors is None and cfg is not None:
            n_max_neighbors = cfg.data.n_max_neighbors or None
        return GNNLibPredictor(model, r_cut=r_cut, n_max_neighbors=n_max_neighbors, device=dev,
                               batch_size=batch_size, progress=progress, source=type(model).__name__)

    from gnn_lib.calculators import CGCNNCalculator  # handles names, checkpoints and the download cache

    calc = CGCNNCalculator(str(model), device=dev)
    pred = GNNLibPredictor.from_calculator(calc, batch_size=batch_size, progress=progress)
    if r_cut is not None:
        pred.r_cut = pred.info["r_cut"] = float(r_cut)
    if n_max_neighbors is not None:
        pred.n_max_neighbors = pred.info["n_max_neighbors"] = n_max_neighbors
    pred.info["source"] = str(model)
    pred.info["model"] = f"gnn_lib ({model})"
    return pred
