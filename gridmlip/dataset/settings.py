import datetime
import importlib.metadata as md
from pathlib import Path

import numpy as np


def save_config(config, output_dir, filename="config.yaml"):
    """Save generation parameters to a YAML file."""
    import yaml

    def to_serializable(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (tuple, list)):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = to_serializable(config)
    config_file = output_dir / filename

    with open(config_file, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    print(f"Configuration saved to: {config_file}")


def load_config(config_file):
    """Load generation parameters from a YAML file."""
    import yaml
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config or {}


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
    path = Path(path)
    record = dict(settings)
    record["versions"] = library_versions()
    record["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_config(record, path.parent, filename=path.name)
    return path


def load_settings(path):
    """Read settings written by save_settings (versions and timestamp included)."""
    return load_config(path)
