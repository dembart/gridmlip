import warnings

import numpy as np
from tqdm import tqdm

PROPERTIES = ("energy", "forces", "stress", "descriptors")


class Predictor:
    """
    Base class of the predictors.

    - ``predictor(atoms_list)`` returns energies, shape (n,), eV; this is what Grid.evaluate uses.
    - ``predictor.predict(atoms_list, properties=(...))`` returns a dict with any of

      ======================  =========================================================
      ``"energy"``            np.ndarray (n,), eV
      ``"forces"``            list of (n_atoms_i, 3) arrays, eV/A
      ``"stress"``            np.ndarray (n, 6), Voigt order xx yy zz yz xz xy, eV/A^3
                              (same convention as ASE's ``atoms.get_stress()``)
      ``"descriptors"``       list of (n_atoms_i, n_features) arrays (per-atom features)
      ======================  =========================================================

    ``predictor.supported`` lists the properties a predictor can return.
    """

    supported = ("energy",)

    def __call__(self, atoms_list):
        return self.predict(atoms_list, ("energy",))["energy"]

    def predict(self, atoms_list, properties=("energy",)):
        properties = self._check(properties)
        if len(atoms_list) == 0:
            return {p: (np.empty(0) if p == "energy" else np.empty((0, 6)) if p == "stress" else [])
                    for p in properties}
        return self._predict(list(atoms_list), properties)

    def _predict(self, atoms_list, properties):
        raise NotImplementedError

    def _check(self, properties):
        if isinstance(properties, str):
            properties = (properties,)
        properties = tuple(dict.fromkeys(properties))
        unknown = [p for p in properties if p not in PROPERTIES]
        if unknown:
            raise ValueError(f"Unknown properties {unknown}; choose from {PROPERTIES}")
        missing = [p for p in properties if p not in self.supported]
        if missing:
            raise NotImplementedError(
                f"{type(self).__name__} cannot return {missing}; supported: {self.supported}")
        return properties


def split_per_atom(array, atoms_list):
    """Split a (sum of n_atoms, ...) array into one array per structure."""
    counts = [len(a) for a in atoms_list]
    return np.split(np.asarray(array), np.cumsum(counts)[:-1])


def to_voigt(stress):
    """(n, 3, 3) or (n, 6) stress -> (n, 6) Voigt (xx yy zz yz xz xy)."""
    stress = np.asarray(stress, dtype=float)
    if stress.ndim == 3:
        return np.stack([stress[:, 0, 0], stress[:, 1, 1], stress[:, 2, 2],
                         stress[:, 1, 2], stress[:, 0, 2], stress[:, 0, 1]], axis=1)
    return stress.reshape(len(stress), 6)


class CalculatorPredictor(Predictor):
    """Any ASE calculator, one structure at a time (slow, but works with everything)."""

    def __init__(self, calc, progress: bool = True):
        self.calc = calc
        self.progress = progress
        self.supported = ("energy", "forces", "stress") + (
            ("descriptors",) if hasattr(calc, "get_descriptors") else ())
        self.info = {"model": f"{type(calc).__module__}.{type(calc).__name__}", "batched": False}

    def _predict(self, atoms_list, properties):
        out = {p: [] for p in properties}
        for atoms in tqdm(atoms_list, desc="getting predictions", disable=not self.progress):
            atoms = atoms.copy()
            atoms.calc = self.calc
            if "energy" in properties:
                out["energy"].append(atoms.get_potential_energy())
            if "forces" in properties:
                out["forces"].append(np.array(atoms.get_forces()))
            if "stress" in properties:
                out["stress"].append(np.array(atoms.get_stress(voigt=True)))
            if "descriptors" in properties:
                out["descriptors"].append(np.asarray(self.calc.get_descriptors(atoms)))
        if "energy" in out:
            out["energy"] = np.array(out["energy"], dtype=float)
        if "stress" in out:
            out["stress"] = np.array(out["stress"], dtype=float).reshape(-1, 6)
        return out


def resolve_device(device: str = "auto") -> str:
    """'auto' -> 'cuda' if available, else 'cpu' (MPS is opt-in: device='mps')."""
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def as_predictor(model):
    """
    Turn a model into a predictor: callable list[Atoms] -> np.ndarray of energies.

    MACE, SevenNet and gnn_lib ASE calculators are recognized and evaluated in batches;
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
            if module.startswith("gnn_lib") and hasattr(model, "model") and hasattr(model, "config"):
                from .gnn_lib import GNNLibPredictor
                return GNNLibPredictor.from_calculator(model)
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
