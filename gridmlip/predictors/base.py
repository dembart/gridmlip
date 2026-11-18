import warnings

import numpy as np
from tqdm import tqdm


class CalculatorPredictor:
    """Any ASE calculator, one structure at a time (slow, but works with everything)."""

    def __init__(self, calc, progress: bool = True):
        self.calc = calc
        self.progress = progress
        self.info = {"model": f"{type(calc).__module__}.{type(calc).__name__}", "batched": False}

    def __call__(self, atoms_list):
        energies = np.empty(len(atoms_list))
        for i, atoms in enumerate(tqdm(atoms_list, desc="getting predictions", disable=not self.progress)):
            atoms = atoms.copy()
            atoms.calc = self.calc
            energies[i] = atoms.get_potential_energy()
        return energies


def resolve_device(device: str = "auto") -> str:
    """'auto' -> 'cuda' if available, else 'cpu' (MPS is opt-in: device='mps')."""
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def as_predictor(model):
    """
    Turn a model into a predictor: callable list[Atoms] -> np.ndarray of energies.

    MACE and SevenNet ASE calculators are recognized and evaluated in batches;
    other ASE calculators are evaluated one structure at a time.
    """
    from ase.calculators.calculator import BaseCalculator

    if isinstance(model, BaseCalculator):
        module = type(model).__module__
        try:
            if module.startswith("mace") and hasattr(model, "models"):
                from .mace import MACEPredictor
                return MACEPredictor.from_calculator(model)
            if module.startswith("sevenn") and hasattr(model, "model"):
                from .sevennet import SevenNetPredictor
                return SevenNetPredictor.from_calculator(model)
        except Exception as exc:  # internals of the library changed: stay correct, just slower
            warnings.warn(f"Batched evaluation of {type(model).__name__} failed to set up ({exc}); "
                          "falling back to one structure at a time.", UserWarning, stacklevel=3)
        return CalculatorPredictor(model)
    if callable(model):
        return model
    raise TypeError(
        f"Cannot evaluate {type(model).__name__}: expected an ASE calculator, a predictor "
        "(callable list[Atoms] -> energies) or a point model such as gridmlip.BVSE."
    )
