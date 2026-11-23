import importlib.util as _util
import warnings as _warnings

from ._version import __version__

if _util.find_spec("numba") is None:
    _warnings.warn(
        "gridmlip: numba is not installed, so percolation analysis uses the scipy backend, "
        "which is 10-100x slower. For the fast numba backend: pip install 'gridmlip[fast]'",
        UserWarning, stacklevel=2,
    )
from .grid import Grid
from .energy_map import EnergyMap
from .path import Path
from .hops import Hop, Hops, HopMap, HopMaps, Paths
from .models import BVSE, bvse
from . import predictors

__all__ = ["Grid", "EnergyMap", "Path", "Hop", "Hops", "HopMap", "HopMaps", "Paths",
           "BVSE", "bvse", "predictors", "__version__"]
