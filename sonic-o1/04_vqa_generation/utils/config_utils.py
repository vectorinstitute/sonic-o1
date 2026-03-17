"""config_utils.py.

Shared configuration loader for VQA generation scripts.

Author: SONIC-O1 Team
"""

from pathlib import Path

import yaml


class Config:
    """Configuration wrapper with nested attribute access."""

    def __init__(self, config_dict):
        """Initialize from nested dict. Recursively wraps dicts as Config."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def load_config(config_path: str, base_dir: Path | None = None) -> Config:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the config file. If relative, resolved
            relative to *base_dir* (or the caller's directory when
            *base_dir* is ``None``).
        base_dir: Optional directory used to resolve relative paths.
            When ``None``, callers should pass
            ``Path(__file__).parent`` explicitly.

    Returns
    -------
        Config object with nested attribute access.
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        if base_dir is None:
            base_dir = Path.cwd()
        config_file = base_dir / config_path

    with open(config_file, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)
