"""
Predictors: callables list[ase.Atoms] -> np.ndarray of energies (eV), same order.

Any such callable can be passed to Grid.evaluate(); ASE calculators are wrapped automatically.
The built-in predictors also have ``predict(atoms_list, properties)`` for energy, forces,
stress and descriptors, computed in the same batched pass.
"""
from .base import PROPERTIES, CalculatorPredictor, Predictor, as_predictor
from .mace import mace, MACEPredictor
from .sevennet import sevennet, SevenNetPredictor
from .gnn_lib import gnn_lib, GNNLibPredictor

__all__ = ["Predictor", "PROPERTIES", "mace", "sevennet", "gnn_lib", "as_predictor", "CalculatorPredictor", "MACEPredictor",
           "SevenNetPredictor", "GNNLibPredictor"]
