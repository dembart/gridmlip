import datetime
import importlib.metadata as md
from pathlib import Path

_PACKAGES = ("gridmlip", "numpy", "scipy", "ase", "spglib", "ions", "fpsample", "dscribe",
             "scikit-learn", "vesin", "numba")


def library_versions():
    """Installed versions of the packages that can change a generated dataset."""
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            if name == "gridmlip":
                from .._version import __version__
                versions[name] = __version__
    return versions


def save_settings(path, **settings):
    """
    Write dataset settings to YAML, together with library versions and a timestamp.

    Example
    -------
    save_settings("dataset/config.yaml", seed=42, specie=3,
                  defects=dict(resolution=0.25, r_min=1.5), rattle=dict(pos_std=0.05))
    """
    from ..utils import save_config

    path = Path(path)
    record = dict(settings)
    record["versions"] = library_versions()
    record["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_config(record, path.parent, filename=path.name)
    return path


def load_settings(path):
    """Read settings written by save_settings (versions and timestamp included)."""
    from ..utils import load_config
    return load_config(path)
