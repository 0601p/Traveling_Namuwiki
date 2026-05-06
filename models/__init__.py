from __future__ import annotations

from utils import Action, Page, Title

from .base import Model
from .randomwalk import RandomWalk


MODEL_REGISTRY: dict[str, type[Model]] = {
    "randomwalk": RandomWalk,
}


def create_model(name: str) -> Model:
    normalized = name.strip().lower().replace("_", "").replace("-", "")
    if normalized not in MODEL_REGISTRY:
        raise NotImplementedError(f"Unknown model: {name}")
    return MODEL_REGISTRY[normalized]()


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
]
