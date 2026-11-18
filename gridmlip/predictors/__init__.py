"""
Predictors: callables list[ase.Atoms] -> np.ndarray of energies (eV), same order.

Any such callable can be passed to Grid.evaluate(); ASE calculators are wrapped automatically.
"""
from .base import CalculatorPredictor, as_predictor
from .mace import mace, MACEPredictor
from .sevennet import sevennet, SevenNetPredictor

__all__ = ["mace", "sevennet", "as_predictor", "CalculatorPredictor", "MACEPredictor", "SevenNetPredictor"]
