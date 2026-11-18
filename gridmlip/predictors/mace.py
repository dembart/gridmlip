from pathlib import Path

import numpy as np
from tqdm import tqdm

from .base import resolve_device


def _import_mace():
    try:
        import torch
        from mace import data
        from mace.tools import torch_geometric, torch_tools, utils
    except ImportError as exc:
        raise ImportError("MACE is required. Install with `pip install 'gridmlip[mace]'`.") from exc
    return torch, data, torch_geometric, torch_tools, utils


class MACEPredictor:
    """
    Batched MACE energies for a list of structures.

    Use ``gridmlip.predictors.mace(...)`` to create one.
    """

    def __init__(self, model, *, head=None, device="auto", dtype="float32", batch_size="auto",
                 progress=True, source=None):
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

    def _predict_batch(self, atoms_list):
        torch, _, torch_geometric, _, _ = _import_mace()
        target = torch.float32 if self.dtype == "float32" else torch.float64
        loader = torch_geometric.dataloader.DataLoader(
            dataset=self._graphs(atoms_list), batch_size=len(atoms_list), shuffle=False, drop_last=False)
        batch = next(iter(loader)).to(self.device)
        for key in batch.keys:
            value = getattr(batch, key)
            if torch.is_tensor(value) and torch.is_floating_point(value):
                setattr(batch, key, value.to(target))
        out = self.model(batch.to_dict(), compute_force=False)
        return out["energy"].detach().cpu().double().numpy()

    def __call__(self, atoms_list):
        torch, *_ = _import_mace()
        if len(atoms_list) == 0:
            return np.empty(0)
        if self.batch_size == "auto":
            batch_size = max(1, 4096 // max(1, len(atoms_list[0])))
        else:
            batch_size = int(self.batch_size)

        energies = []
        start = 0
        with tqdm(total=len(atoms_list), desc="getting predictions", disable=not self.progress) as bar:
            while start < len(atoms_list):
                chunk = atoms_list[start:start + batch_size]
                try:
                    energies.append(self._predict_batch(chunk))
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
        return np.concatenate(energies)


def mace(model, *, head=None, device="auto", dtype="float32", batch_size="auto", progress=True):
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

    Returns
    -------
    MACEPredictor, callable list[Atoms] -> energies
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
                         progress=progress, source=source)
