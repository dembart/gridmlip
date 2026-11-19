"""
Building blocks for MLIP training sets:

    cfgs = interstitials(...) + vacancies(...) + frenkel_pairs(...)   # gridmlip.defects
    X    = soap_features(cfgs)
    idx  = select_fps(X, max_samples=500)
    data = [rattle(cfgs[j], seed=42) for j in idx]

or all of it in one call (also from the shell: ``gridmlip dataset structures.xyz --config dataset.yaml``):

    build_dataset("structures.xyz", config="dataset.yaml", output_dir="datasets/li")
"""
from .features import soap_features
from .selection import select_fps
from .rattle import rattle, rattle_rng
from .settings import save_settings, load_settings, library_versions
from .recipe import build_dataset

__all__ = ["build_dataset", "soap_features", "select_fps", "rattle", "rattle_rng", "save_settings", "load_settings",
           "library_versions"]
