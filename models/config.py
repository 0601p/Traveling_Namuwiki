from __future__ import annotations

from pathlib import Path
from typing import Mapping

from embed import load_embedding_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "model"
DEFAULT_LINEAR_CONFIG = CONFIG_DIR / "linear.yaml"


def load_model_config(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return load_embedding_config(path)


def config_value(
    config: Mapping[str, object],
    key: str,
    default: object = None,
) -> object:
    value = config.get(key, default)
    return default if value is None else value
