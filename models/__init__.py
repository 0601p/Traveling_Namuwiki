from __future__ import annotations

import warnings

from utils import Action, Page, Title

from .base import Model
from .randomwalk import RandomWalk


MODEL_REGISTRY: dict[str, type[Model]] = {
    "randomwalk": RandomWalk,
}
_OPTIONAL_EXPORTS: list[str] = []

try:
    from .ar_walk import AutoregressiveWalk
except ImportError as exc:
    warnings.warn(
        "Skipping ar_walk registration because optional dependencies are missing: "
        f"{exc}",
        stacklevel=1,
    )
else:
    MODEL_REGISTRY["ar_walk"] = AutoregressiveWalk
    _OPTIONAL_EXPORTS.append("AutoregressiveWalk")


def create_model(name: str) -> Model:
    normalized = name.strip().lower().replace("_", "").replace("-", "")
    normalized_registry = {
        key.replace("_", "").replace("-", ""): model_class
        for key, model_class in MODEL_REGISTRY.items()
    }
    if normalized not in normalized_registry:
        raise NotImplementedError(f"Unknown model: {name}")
    return normalized_registry[normalized]()


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


__all__ = [
    "Action",
    "Model",
    "Page",
    "RandomWalk",
    "Title",
    "available_models",
    "create_model",
    *_OPTIONAL_EXPORTS,
]
