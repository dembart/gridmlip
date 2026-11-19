"""
gnn_lib predictor, against a stand-in with gnn_lib's interface (atoms2data, collate_fn, a CGCNN-like
model with per_atom_target and per-element reference energies, CGCNNCalculator with .model/.config).
The real library is checked separately (it is not on PyPI).
"""
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gridmlip import Grid  # noqa: E402
from structures import nacl  # noqa: E402

E0 = {11: -1.5, 17: -3.0}       # per-element reference energies of the stand-in model


class _Data:
    def __init__(self, values, n_atoms, numbers):
        self.values, self.num_atoms, self.numbers = values, n_atoms, numbers
        self.batch = torch.repeat_interleave(torch.arange(len(n_atoms)), n_atoms)

    def to(self, device):
        return self


class _Model(torch.nn.Module):
    """Per-atom residual energy = sum|r| / n_atoms (like CGCNN with per_atom_target=True)."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.per_atom_target = True
        self.compute_forces = self.compute_stress = False
        ref = torch.zeros(100)
        for z, e in E0.items():
            ref[z] = e
        self.register_buffer("ref_energies", ref)

    def _energy_from_e0s_and_residuals(self, energy, data):   # same logic as gnn_lib
        ref_sum = torch.zeros(len(data.num_atoms)).index_add_(0, data.batch, self.ref_energies[data.numbers])
        return energy + (ref_sum / data.num_atoms).unsqueeze(-1)

    def forward(self, data):
        n = data.num_atoms
        out = {"energy": (data.values * self.scale / n).unsqueeze(-1), "forces": None, "stress": None}
        if self.compute_forces:
            out["forces"] = torch.ones(int(n.sum()), 3)
        if self.compute_stress:
            out["stress"] = torch.eye(3).repeat(len(n), 1, 1)
        return out


class _Cfg:
    class data:
        r_cut, n_max_neighbors = 6.0, 12


SEEN = {}


@pytest.fixture
def fake_gnn_lib(monkeypatch):
    mods = {k: types.ModuleType(k) for k in ("gnn_lib", "gnn_lib.graph_data", "gnn_lib.data_utils", "gnn_lib.calculators")}
    mods["gnn_lib"].__path__ = []   # make it a package

    def atoms2data(atoms, r_cut, n_max_neighbors):
        SEEN["r_cut"], SEEN["n_max_neighbors"] = r_cut, n_max_neighbors
        return (float(np.linalg.norm(atoms.positions, axis=1).sum()), len(atoms), atoms.numbers)

    def collate_fn(items):
        return _Data(torch.tensor([v for v, _, _ in items], dtype=torch.float32),
                     torch.tensor([n for _, n, _ in items]),
                     torch.as_tensor(np.concatenate([z for _, _, z in items])))

    from ase.calculators.calculator import Calculator

    class CGCNNCalculator(Calculator):   # gnn_lib's calculator is an ASE Calculator too
        def __init__(self, model, device="cuda"):
            super().__init__()
            SEEN["calc_model"] = model
            self.model, self.config, self.device = _Model(), _Cfg, device

    CGCNNCalculator.__module__ = "gnn_lib.calculators.cgcnn_calculator"
    mods["gnn_lib.graph_data"].atoms2data = atoms2data
    mods["gnn_lib.data_utils"].collate_fn = collate_fn
    mods["gnn_lib.calculators"].CGCNNCalculator = CGCNNCalculator
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return CGCNNCalculator


def _expected(cfgs):  # what CGCNNCalculator would give: (residual per atom + e0 per atom) * n_atoms
    return np.array([np.linalg.norm(c.positions, axis=1).sum() + sum(E0[z] for z in c.numbers) for c in cfgs])


@pytest.fixture
def grid():
    return Grid(nacl(), "Na", resolution=0.5, r_min=1.2)


def test_total_energies_match_calculator_convention(fake_gnn_lib, grid):
    from gridmlip.predictors import gnn_lib

    cfgs = grid.prepare_configurations()
    pred = gnn_lib("matpes_pbe_small", device="cpu", batch_size=7, progress=False)
    assert SEEN["calc_model"] == "matpes_pbe_small"
    np.testing.assert_allclose(pred(cfgs), _expected(cfgs), rtol=1e-5)
    assert SEEN["r_cut"] == 6.0 and SEEN["n_max_neighbors"] == 12          # taken from the model config
    emap = grid.evaluate(pred)
    np.testing.assert_allclose(emap.values, grid.load(_expected(cfgs)).values, rtol=1e-5)


def test_calculator_is_batched_automatically(fake_gnn_lib, grid):
    from gridmlip.predictors import GNNLibPredictor, as_predictor

    pred = as_predictor(fake_gnn_lib("matpes_pbe_small", device="cpu"))
    assert isinstance(pred, GNNLibPredictor) and pred.r_cut == 6.0


def test_forces_and_stress_batched(fake_gnn_lib, grid):
    from gridmlip.predictors import gnn_lib

    cfgs = grid.prepare_configurations()[:9]
    pred = gnn_lib("matpes_pbe_small", device="cpu", batch_size=4, progress=False)
    out = pred.predict(cfgs, ("energy", "forces", "stress"))
    assert [f.shape for f in out["forces"]] == [(len(c), 3) for c in cfgs]
    np.testing.assert_array_equal(out["stress"], np.tile([1, 1, 1, 0, 0, 0], (9, 1)))
    assert not pred.model.compute_forces and not pred.model.compute_stress   # restored after the call
    with pytest.raises(NotImplementedError, match="descriptors"):
        pred.predict(cfgs, ("descriptors",))
