"""
Building blocks for MLIP training sets:

    cfgs = interstitials(...) + vacancies(...) + frenkel_pairs(...)   # gridmlip.defects
    X    = soap_features(cfgs)
    idx  = select_fps(X, max_samples=500)
    data = [rattle(cfgs[j], seed=42) for j in idx]

or, from the shell: ``gridmlip dataset structures.xyz --config dataset.yaml``.
"""
from .features import soap_features
from .selection import select_fps
from .rattle import rattle, rattle_rng
from .settings import save_settings, load_settings, library_versions
from ._legacy import generate_defects_dataset

__all__ = ["soap_features", "select_fps", "rattle", "rattle_rng", "save_settings", "load_settings",
           "library_versions", "generate_defects_dataset"]
