"""
gnn_lib predictor. gnn_lib itself is not installed in the test environment, so a stand-in with the
same interface (atoms2data, InMemoryDataset.from_data_list, collate_fn, model(batch)['energy'])
checks the plumbing: batching, order, total energies, settings passed to graph construction.
"""
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gridmlip import Grid  # noqa: E402
from structures import nacl  # noqa: E402

SEEN = {}


class _Batch:
    def __init__(self, values):
        self.values = values

    def to(self, device):
        return self


class _Model(torch.nn.Module):
    """'energy' = sum over atoms of |position| (a deterministic toy)."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        assert self.per_atom_target is False and self.compute_forces is False
        return {"energy": (batch.values * self.scale).unsqueeze(-1)}  # shape (n, 1) like many GNNs


@pytest.fixture
def fake_gnn_lib(monkeypatch):
    pkg = types.ModuleType("gnn_lib")
    graph_data = types.ModuleType("gnn_lib.graph_data")
    data_utils = types.ModuleType("gnn_lib.data_utils")

    def atoms2data(atoms, r_cut, n_max_neighbors):
        SEEN["r_cut"], SEEN["n_max_neighbors"] = r_cut, n_max_neighbors
        return float(np.linalg.norm(atoms.positions, axis=1).sum())

    class InMemoryDataset(list):
        @classmethod
        def from_data_list(cls, data):
            return cls(data)

    graph_data.atoms2data = atoms2data
    graph_data.InMemoryDataset = InMemoryDataset
    data_utils.collate_fn = lambda items: _Batch(torch.tensor(items, dtype=torch.float32))
    for name, mod in [("gnn_lib", pkg), ("gnn_lib.graph_data", graph_data), ("gnn_lib.data_utils", data_utils)]:
        monkeypatch.setitem(sys.modules, name, mod)


def _expected(cfgs):
    return np.array([np.linalg.norm(c.positions, axis=1).sum() for c in cfgs])


def test_gnn_lib_predictor_batches_in_order(fake_gnn_lib, tmp_path):
    from gridmlip.predictors import GNNLibPredictor, gnn_lib

    grid = Grid(nacl(), "Na", resolution=0.5, r_min=1.2)
    cfgs = grid.prepare_configurations()
    torch.save(_Model(), tmp_path / "model.pth")

    pred = gnn_lib(tmp_path / "model.pth", r_cut=6.0, n_max_neighbors=12, device="cpu",
                   batch_size=7, progress=False)
    assert isinstance(pred, GNNLibPredictor)
    np.testing.assert_allclose(pred(cfgs), _expected(cfgs), rtol=1e-5)
    assert SEEN == {"r_cut": 6.0, "n_max_neighbors": 12}

    emap = grid.evaluate(pred)
    np.testing.assert_allclose(emap.values, grid.load_energies(_expected(cfgs)).values, rtol=1e-5)
    assert emap.info["model"].startswith("gnn_lib") and emap.info["r_cut"] == 6.0


def test_gnn_lib_missing_file(fake_gnn_lib):
    from gridmlip.predictors import gnn_lib

    with pytest.raises(ValueError, match="not an existing model file"):
        gnn_lib("no-such-model.pth")
