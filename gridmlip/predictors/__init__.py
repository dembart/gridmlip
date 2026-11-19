"""
Predictors: callables list[ase.Atoms] -> np.ndarray of energies (eV), same order.

Any such callable can be passed to Grid.evaluate(); ASE calculators are wrapped automatically.
"""
from .base import CalculatorPredictor, as_predictor
from .mace import mace, MACEPredictor
from .sevennet import sevennet, SevenNetPredictor
from .gnn_lib import gnn_lib, GNNLibPredictor

__all__ = ["mace", "sevennet", "gnn_lib", "as_predictor", "CalculatorPredictor", "MACEPredictor",
           "SevenNetPredictor", "GNNLibPredictor"]
