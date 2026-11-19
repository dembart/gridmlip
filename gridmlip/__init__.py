import importlib.util as _util
import warnings as _warnings

from ._version import __version__

if _util.find_spec("numba") is None:
    _warnings.warn(
        "gridmlip: numba is not installed, so percolation analysis uses the scipy backend, "
        "which is 10-100x slower. For the fast numba backend: pip install 'gridmlip[fast]'",
        UserWarning, stacklevel=2,
    )
from .core import Grid
from .energy_map import EnergyMap
from .models import BVSE, bvse
from . import predictors

__all__ = ["Grid", "EnergyMap", "BVSE", "bvse", "predictors", "__version__"]
