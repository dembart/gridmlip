import numpy as np
from tqdm import tqdm

from .base import resolve_device


class SevenNetPredictor:
    """
    Batched SevenNet energies for a list of structures.

    Use ``gridmlip.predictors.sevennet(...)`` to create one.

    Note: written against the SevenNet API used by gridmlip <= 0.2.7
    (``sevenn.train.dataload.graph_build``); not covered by the test suite.
    """

    def __init__(self, model, cutoff, *, modal=None, device="cpu", batch_size=64, num_workers=1,
                 progress=True, source=None):
        self.model = model.to(device)
        self.model.eval()
        self.model.set_is_batch_data(True)
        self.cutoff = cutoff
        self.modal = modal
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.progress = progress
        self.info = {"model": f"SevenNet ({source})" if source else "SevenNet", "source": source,
                     "modal": modal, "device": str(device), "batched": True}

    @classmethod
    def from_calculator(cls, calc, batch_size=64, progress=True):
        """Batched predictor from an existing SevenNetCalculator (same model, modal and device)."""
        return cls(calc.model, calc.cutoff, modal=getattr(calc, "modal", None), device=calc.device,
                   batch_size=batch_size, progress=progress, source=type(calc).__name__)

    def __call__(self, atoms_list):
        try:
            from torch_geometric.loader import DataLoader
            from sevenn.train import dataload
            import sevenn.util as util
            import sevenn._keys as KEY
        except ImportError as exc:
            raise ImportError("SevenNet is required. Install with `pip install 'gridmlip[sevenn]'`.") from exc

        graphs = dataload.graph_build(atoms_list, self.cutoff, allow_unlabeled=True,
                                      y_from_calc=True, num_cores=self.num_workers)
        if self.modal is not None:
            for g in graphs:
                g[KEY.DATA_MODALITY] = self.modal

        energies = []
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        for batch in tqdm(loader, desc="getting predictions", disable=not self.progress):
            out = self.model(batch.to(self.device)).detach().cpu()
            for g in util.to_atom_graph_list(out):
                energies.append(float(g.fit_dimension()[KEY.PRED_TOTAL_ENERGY]))
        return np.array(energies)


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
    SevenNetPredictor, callable list[Atoms] -> energies
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
