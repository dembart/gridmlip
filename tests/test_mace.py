"""
MACE predictor. Needs mace-torch and a model:
  - foundation model 'small' from the mace cache (or network),
  - optionally a multi-head model: set GRIDMLIP_TEST_MACE_MULTIHEAD=/path/to/model.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.umlip  # slow: run with `pytest -m umlip`

mace_calculators = pytest.importorskip("mace.calculators")

import gridmlip as gm  # noqa: E402
from gridmlip.predictors import MACEPredictor, as_predictor, mace  # noqa: E402

from structures import licoo2  # noqa: E402


@pytest.fixture(scope="module")
def grid():
    return gm.Grid(licoo2(), "Li", resolution=0.5, r_cut=5.0, r_min=1.2)


@pytest.fixture(scope="module")
def calc():
    try:
        return mace_calculators.mace_mp(model="small", device="cpu", default_dtype="float64")
    except Exception as exc:  # no cache and no network
        pytest.skip(f"MACE-MP small not available: {exc}")


def _one_by_one(calc, cfgs):
    out = []
    for c in cfgs:
        c = c.copy()
        c.calc = calc
        out.append(c.get_potential_energy())
    return np.array(out)


def test_batched_equals_calculator(grid, calc):
    cfgs = grid.prepare_configurations()[:40]
    pred = as_predictor(calc)
    assert isinstance(pred, MACEPredictor)          # recognized and batched
    pred.batch_size = 7                              # several batches, one partial
    np.testing.assert_allclose(pred(cfgs), _one_by_one(calc, cfgs), rtol=0, atol=1e-6)


def test_mace_by_name_float32(grid, calc):
    pred = mace("small", device="cpu")
    assert pred.info["dtype"] == "float32" and pred.info["head"] == "Default"
    cfgs = grid.prepare_configurations()[:20]
    # float32 vs float64: differences well below meV per structure
    np.testing.assert_allclose(pred(cfgs), _one_by_one(calc, cfgs), rtol=0, atol=5e-3)


def test_evaluate_with_calculator(grid, calc):
    emap = grid.evaluate(calc)
    assert emap.info["batched"] is True and emap.info["dtype"] == "float64"
    assert np.isfinite(emap.values).any()


def test_unknown_name_is_reported():
    with pytest.raises(ValueError, match="neither an existing file"):
        mace("definitely-not-a-model")


@pytest.mark.skipif(not os.environ.get("GRIDMLIP_TEST_MACE_MULTIHEAD"), reason="no multi-head model given")
def test_multihead_requires_head(grid):
    path = os.environ["GRIDMLIP_TEST_MACE_MULTIHEAD"]
    with pytest.raises(ValueError, match="several heads"):
        mace(path, device="cpu", dtype="float64")
    import torch
    heads = list(torch.load(path, map_location="cpu", weights_only=False).heads)
    cfgs = grid.prepare_configurations()[:5]
    # this model contains TorchScript parts stored in float64, so it cannot be cast to float32
    e0 = mace(path, head=heads[0], device="cpu", dtype="float64")(cfgs)
    e1 = mace(path, head=heads[-1], device="cpu", dtype="float64")(cfgs)
    assert e0.shape == e1.shape == (5,)
    assert not np.allclose(e0, e1)  # heads give different energies


def test_batched_forces_stress_descriptors_equal_calculator(grid, calc):
    cfgs = grid.prepare_configurations()[:12]
    pred = as_predictor(calc)
    pred.batch_size = 5                              # several batches, one partial
    out = pred.predict(cfgs, ("energy", "forces", "stress", "descriptors"))
    assert set(out) == {"energy", "forces", "stress", "descriptors"}
    assert out["stress"].shape == (12, 6) and len(out["forces"]) == len(out["descriptors"]) == 12
    for i, c in enumerate(cfgs):
        c = c.copy()
        c.calc = calc
        np.testing.assert_allclose(out["energy"][i], c.get_potential_energy(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(out["forces"][i], c.get_forces(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(out["stress"][i], c.get_stress(), rtol=0, atol=1e-7)
        np.testing.assert_allclose(out["descriptors"][i], calc.get_descriptors(c), rtol=0, atol=1e-6)


def test_energy_only_call_matches_predict(grid, calc):
    cfgs = grid.prepare_configurations()[:6]
    pred = as_predictor(calc)
    np.testing.assert_array_equal(pred(cfgs), pred.predict(cfgs)["energy"])
