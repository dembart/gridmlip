import yaml
from pathlib import Path
import numpy as np



def save_config(config, output_dir, filename="config.yaml"):
    """Save generation parameters to a YAML file."""

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
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config or {}