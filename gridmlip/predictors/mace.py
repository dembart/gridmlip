from pathlib import Path

import numpy as np
from tqdm import tqdm

from .base import Predictor, resolve_device, split_per_atom, to_voigt


def _import_mace():
    try:
        import torch
        from mace import data
        from mace.tools import torch_geometric, torch_tools, utils
    except ImportError as exc:
        raise ImportError("MACE is required. Install it with `pip install mace-torch`.") from exc
    return torch, data, torch_geometric, torch_tools, utils


class MACEPredictor(Predictor):
    """
    Batched MACE predictions for a list of structures.

    ``predictor(atoms_list)`` -> energies; ``predictor.predict(atoms_list, properties)`` -> energy,
    forces, stress and/or descriptors (see gridmlip.predictors.base.Predictor).
    Use ``gridmlip.predictors.mace(...)`` to create one.
    """

    supported = ("energy", "forces", "stress", "descriptors")

    def __init__(self, model, *, head=None, device="auto", dtype="float32", batch_size="auto",
                 progress=True, source=None, invariants_only=True, descriptor_layers=-1):
        torch, *_ = _import_mace()
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        self.device = torch.device(resolve_device(device))
        self.dtype = dtype
        self.model = (model.float() if dtype == "float32" else model.double()).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.heads = [str(h) for h in (getattr(self.model, "heads", None) or ["Default"])]
        if head is None:
            if len(self.heads) > 1:
                raise ValueError(f"This MACE model has several heads, choose one with head=...: {self.heads}")
            head = self.heads[0]
        elif head not in self.heads:
            raise ValueError(f"Unknown head {head!r}; available heads: {self.heads}")
        self.head = head
        self.batch_size = batch_size
        self.progress = progress
        self.invariants_only = invariants_only
        self.descriptor_layers = descriptor_layers
        self.info = {
            "model": f"MACE ({source})" if source else "MACE",
            "source": source, "head": head, "dtype": dtype, "device": str(self.device), "batched": True,
        }

    @classmethod
    def from_calculator(cls, calc, batch_size="auto", progress=True):
        """Batched predictor from an existing MACECalculator (same model, device and dtype)."""
        if len(calc.models) != 1:
            raise ValueError("committee MACE calculators are not supported for batching")
        model = calc.models[0]
        dtype = "float64" if next(model.parameters()).dtype == __import__("torch").float64 else "float32"
        heads = [str(h) for h in (getattr(model, "heads", None) or ["Default"])]
        head = getattr(calc, "head", None)
        if head is None:  # what MACECalculator uses when atoms.info has no 'head'
            head = "Default" if "Default" in heads else heads[-1]
        return cls(model, head=head, device=str(calc.device), dtype=dtype, batch_size=batch_size,
                   progress=progress, source=f"{type(calc).__name__}")

    def _graphs(self, atoms_list):
        _, data, _, _, utils = _import_mace()
        z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        cutoff = float(self.model.r_max)
        graphs = []
        for atoms in atoms_list:
            config = data.config_from_atoms(atoms)
            config.head = self.head
            graphs.append(data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff,
                                                      heads=self.heads))
        return graphs

    def _descriptors(self, node_feats):
        """Per-atom descriptors as in MACECalculator.get_descriptors."""
        from mace.modules.utils import extract_invariant

        num_layers = int(self.model.num_interactions) if self.descriptor_layers == -1 else self.descriptor_layers
        if not self.invariants_only:
            return node_feats
        irreps_out = self.model.products[0].linear.__dict__["irreps_out"]
        l_max = irreps_out.lmax
        num_features = irreps_out.dim // (l_max + 1) ** 2
        return extract_invariant(node_feats, num_layers=num_layers, num_features=num_features, l_max=l_max)

    def _predict_batch(self, atoms_list, properties):
        torch, _, torch_geometric, _, _ = _import_mace()
        target = torch.float32 if self.dtype == "float32" else torch.float64
        loader = torch_geometric.dataloader.DataLoader(
            dataset=self._graphs(atoms_list), batch_size=len(atoms_list), shuffle=False, drop_last=False)
        batch = next(iter(loader)).to(self.device)
        for key in batch.keys:
            value = getattr(batch, key)
            if torch.is_tensor(value) and torch.is_floating_point(value):
                setattr(batch, key, value.to(target))
        out = self.model(batch.to_dict(), compute_force="forces" in properties,
                         compute_stress="stress" in properties)

        def np_(t):
            return t.detach().cpu().double().numpy()

        res = {}
        if "energy" in properties:
            res["energy"] = np_(out["energy"]).reshape(-1)
        if "forces" in properties:
            res["forces"] = split_per_atom(np_(out["forces"]), atoms_list)
        if "stress" in properties:
            res["stress"] = to_voigt(np_(out["stress"]))
        if "descriptors" in properties:
            res["descriptors"] = split_per_atom(np_(self._descriptors(out["node_feats"])), atoms_list)
        return res

    def _predict(self, atoms_list, properties):
        torch, *_ = _import_mace()
        if self.batch_size == "auto":
            per_atom = 4096 if properties == ("energy",) else 2048  # gradients need more memory
            batch_size = max(1, per_atom // max(1, len(atoms_list[0])))
        else:
            batch_size = int(self.batch_size)

        parts = []
        start = 0
        with tqdm(total=len(atoms_list), desc="getting predictions", disable=not self.progress) as bar:
            while start < len(atoms_list):
                chunk = atoms_list[start:start + batch_size]
                try:
                    parts.append(self._predict_batch(chunk, properties))
                except RuntimeError as exc:  # CUDA out of memory: halve the batch and retry
                    if "out of memory" in str(exc).lower() and batch_size > 1:
                        batch_size //= 2
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    if "same dtype" in str(exc):
                        other = "float64" if self.dtype == "float32" else "float32"
                        raise RuntimeError(
                            f"This model cannot run in {self.dtype} (it has parts with a fixed dtype); "
                            f"use dtype={other!r}."
                        ) from exc
                    raise
                start += len(chunk)
                bar.update(len(chunk))

        res = {}
        for p in properties:
            if p in ("forces", "descriptors"):
                res[p] = [a for part in parts for a in part[p]]
            else:
                res[p] = np.concatenate([part[p] for part in parts])
        return res


def mace(model, *, head=None, device="auto", dtype="float32", batch_size="auto", progress=True,
         invariants_only=True, descriptor_layers=-1):
    """
    Batched MACE predictor.

    Parameters
    ----------
    model: str, path or torch model
        path to a .model file, or a foundation-model name understood by
        ``mace.calculators.mace_mp`` (e.g. 'small', 'medium', 'medium-mpa-0')

    head: str, optional
        required for multi-head models (an error lists the available heads)

    device: str, 'auto' by default
        'auto' (cuda if available, else cpu), 'cuda', 'cpu', 'mps'

    dtype: str, 'float32' by default
        'float32' or 'float64'

    batch_size: int or 'auto'
        'auto' uses ~4096 atoms per batch and halves it on CUDA out-of-memory errors

    progress: bool, True by default
        show a progress bar

    invariants_only, descriptor_layers:
        descriptors as in MACECalculator.get_descriptors (invariant features of all layers by default)

    Returns
    -------
    MACEPredictor: ``pred(atoms_list)`` -> energies;
    ``pred.predict(atoms_list, ("energy", "forces", "stress", "descriptors"))`` -> dict
    """
    torch, *_ = _import_mace()
    dev = resolve_device(device)
    if isinstance(model, torch.nn.Module):
        module, source = model, type(model).__name__
    elif Path(str(model)).expanduser().is_file():
        path = Path(str(model)).expanduser()
        module, source = torch.load(path, map_location=dev, weights_only=False), str(path)
    else:
        from mace.calculators import mace_mp
        try:
            module = mace_mp(model=str(model), device=dev, default_dtype=dtype, return_raw_model=True)
        except Exception as exc:
            raise ValueError(
                f"{model!r} is neither an existing file nor a model name known to mace_mp ({exc})"
            ) from exc
        source = str(model)
    return MACEPredictor(module, head=head, device=dev, dtype=dtype, batch_size=batch_size,
                         progress=progress, source=source, invariants_only=invariants_only,
                         descriptor_layers=descriptor_layers)
