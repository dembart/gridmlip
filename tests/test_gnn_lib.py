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
    def __init__(self, values, n_atoms):
        self.values = values
        self.n_atoms = n_atoms

    def to(self, device):
        return self


class _Model(torch.nn.Module):
    """'energy' = sum over atoms of |position| (a deterministic toy)."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        assert self.per_atom_target is False
        n = len(batch.values)
        out = {"energy": (batch.values * self.scale).unsqueeze(-1),       # shape (n, 1) like many GNNs
               "node_feats": torch.arange(4 * batch.n_atoms.sum(), dtype=torch.float32).reshape(-1, 4)}
        if self.compute_forces:
            out["forces"] = torch.ones(int(batch.n_atoms.sum()), 3)
        if self.compute_stress:
            out["stress"] = torch.eye(3).repeat(n, 1, 1)
        return out


@pytest.fixture
def fake_gnn_lib(monkeypatch):
    pkg = types.ModuleType("gnn_lib")
    graph_data = types.ModuleType("gnn_lib.graph_data")
    data_utils = types.ModuleType("gnn_lib.data_utils")

    def atoms2data(atoms, r_cut, n_max_neighbors):
        SEEN["r_cut"], SEEN["n_max_neighbors"] = r_cut, n_max_neighbors
        return (float(np.linalg.norm(atoms.positions, axis=1).sum()), len(atoms))

    class InMemoryDataset(list):
        @classmethod
        def from_data_list(cls, data):
            return cls(data)

    graph_data.atoms2data = atoms2data
    graph_data.InMemoryDataset = InMemoryDataset
    data_utils.collate_fn = lambda items: _Batch(torch.tensor([v for v, _ in items], dtype=torch.float32),
                                                 torch.tensor([n for _, n in items]))
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
    np.testing.assert_allclose(emap.values, grid.load(_expected(cfgs)).values, rtol=1e-5)
    assert emap.info["model"].startswith("gnn_lib") and emap.info["r_cut"] == 6.0


def test_gnn_lib_missing_file(fake_gnn_lib):
    from gridmlip.predictors import gnn_lib

    with pytest.raises(ValueError, match="not an existing model file"):
        gnn_lib("no-such-model.pth")


def test_gnn_lib_all_properties_batched(fake_gnn_lib):
    from gridmlip.predictors import gnn_lib

    grid = Grid(nacl(), "Na", resolution=0.5, r_min=1.2)
    cfgs = grid.prepare_configurations()[:9]
    pred = gnn_lib(_Model(), device="cpu", batch_size=4, progress=False)
    out = pred.predict(cfgs, ("energy", "forces", "stress", "descriptors"))
    np.testing.assert_allclose(out["energy"], _expected(cfgs), rtol=1e-5)
    assert [f.shape for f in out["forces"]] == [(len(c), 3) for c in cfgs]
    np.testing.assert_array_equal(out["stress"], np.tile([1, 1, 1, 0, 0, 0], (9, 1)))
    assert [d.shape for d in out["descriptors"]] == [(len(c), 4) for c in cfgs]
    assert pred.model.compute_forces and pred.model.compute_stress
    pred(cfgs)  # energy-only call switches them off again
    assert not pred.model.compute_forces and not pred.model.compute_stress
