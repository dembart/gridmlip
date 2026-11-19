import numpy as np
from tqdm import tqdm

from .base import Predictor, resolve_device, split_per_atom


def _import_sevenn():
    try:
        from torch_geometric.loader import DataLoader
        from sevenn.atom_graph_data import AtomGraphData
        from sevenn.train.dataload import unlabeled_atoms_to_graph
        import sevenn._keys as KEY
    except ImportError as exc:
        raise ImportError("SevenNet is required. Install with `pip install 'gridmlip[sevenn]'`.") from exc

    def to_graph(atoms, cutoff):  # same graph construction as SevenNetCalculator
        return AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms, cutoff, with_shift=False))
    return DataLoader, to_graph, KEY


class SevenNetPredictor(Predictor):
    """
    Batched SevenNet predictions for a list of structures.

    ``predictor(atoms_list)`` -> energies; ``predictor.predict(atoms_list, properties)`` -> energy,
    forces, stress and/or descriptors, all from the same batched forward pass.
    Use ``gridmlip.predictors.sevennet(...)`` to create one.

    Conventions follow SevenNetCalculator (sevenn 0.13): stress = -PRED_STRESS reordered to
    ASE Voigt order. Descriptors are the per-atom features entering the energy readout
    (output of the last interaction block). Not covered by the test suite (sevenn not installed).
    """

    supported = ("energy", "forces", "stress", "descriptors")

    def __init__(self, model, cutoff, *, modal=None, device="cpu", batch_size=64,
                 progress=True, source=None):
        self.model = model.to(device)
        self.model.eval()
        self.model.set_is_batch_data(True)
        self.cutoff = cutoff
        self.modal = modal
        self.device = device
        self.batch_size = batch_size
        self.progress = progress
        self.info = {"model": f"SevenNet ({source})" if source else "SevenNet", "source": source,
                     "modal": modal, "device": str(device), "batched": True}

    @classmethod
    def from_calculator(cls, calc, batch_size=64, progress=True):
        """Batched predictor from an existing SevenNetCalculator (same model, modal and device)."""
        return cls(calc.model, calc.cutoff, modal=getattr(calc, "modal", None), device=calc.device,
                   batch_size=batch_size, progress=progress, source=type(calc).__name__)

    def _readout_hook(self, store):
        """Capture node features right before the energy readout."""
        module = dict(self.model.named_children()).get("reduce_input_to_hidden")
        if module is None:
            raise NotImplementedError("descriptors: this SevenNet model has no 'reduce_input_to_hidden' block")

        def hook(_module, args):
            store.append(args[0]["x"].detach().cpu().double().numpy())
        return module.register_forward_pre_hook(hook)

    def _predict(self, atoms_list, properties):
        DataLoader, to_graph, KEY = _import_sevenn()
        res = {p: [] for p in properties}
        with tqdm(total=len(atoms_list), desc="getting predictions", disable=not self.progress) as bar:
            # graphs are built per chunk, so memory does not grow with the grid size
            for start in range(0, len(atoms_list), self.batch_size):
                chunk = atoms_list[start:start + self.batch_size]
                graphs = [to_graph(a, self.cutoff) for a in chunk]
                if self.modal is not None:
                    for g in graphs:
                        g[KEY.DATA_MODALITY] = self.modal
                batch = next(iter(DataLoader(graphs, batch_size=len(graphs), shuffle=False)))

                features = []
                handle = self._readout_hook(features) if "descriptors" in properties else None
                try:
                    out = self.model(batch.to(self.device))
                finally:
                    if handle is not None:
                        handle.remove()

                if "energy" in res:
                    res["energy"].append(out[KEY.PRED_TOTAL_ENERGY].detach().cpu().double().numpy().reshape(-1))
                if "forces" in res:
                    res["forces"] += split_per_atom(out[KEY.PRED_FORCE].detach().cpu().double().numpy(), chunk)
                if "stress" in res:
                    if KEY.PRED_STRESS not in out:
                        raise NotImplementedError("this SevenNet model does not predict stress")
                    s = -out[KEY.PRED_STRESS].detach().cpu().double().numpy().reshape(len(chunk), 6)
                    res["stress"].append(s[:, [0, 1, 2, 4, 5, 3]])
                if "descriptors" in res:
                    res["descriptors"] += split_per_atom(features[0], chunk)
                bar.update(len(chunk))

        for p in ("energy", "stress"):
            if p in res:
                res[p] = np.concatenate(res[p])
        return res


def sevennet(model, *, modal=None, device="auto", batch_size=64, progress=True):
    """
    Batched SevenNet predictor.

    Parameters
    ----------
    model: str
        path to a checkpoint, or a pretrained name known to SevenNet (e.g. '7net-0', '7net-mf-ompa')

    modal: str, optional
        required for multi-fidelity models (e.g. 'mpa', 'omat24')

    device: str, 'auto' by default

    batch_size: int, 64 by default

    Returns
    -------
    SevenNetPredictor: ``pred(atoms_list)`` -> energies;
    ``pred.predict(atoms_list, ("energy", "forces", "stress", "descriptors"))`` -> dict
    """
    try:
        from sevenn.calculator import SevenNetCalculator
    except ImportError as exc:
        raise ImportError("SevenNet is required. Install with `pip install 'gridmlip[sevenn]'`.") from exc
    kwargs = {"modal": modal} if modal is not None else {}
    calc = SevenNetCalculator(model=str(model), device=resolve_device(device), **kwargs)
    pred = SevenNetPredictor.from_calculator(calc, batch_size=batch_size, progress=progress)
    pred.info["source"] = str(model)
    pred.info["model"] = f"SevenNet ({model})"
    return pred
