from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from utils import Action, Page, Title

from .base import Model
from .linear import LinearModel
from .randomwalk import RandomWalk
from .semantic_walk import SemanticWalk


MODEL_REGISTRY: dict[str, type[Model]] = {
    "randomwalk": RandomWalk,
    "linear": LinearModel,
    "semanticwalk": SemanticWalk,
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


def normalize_model_name(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "")


def normalized_registry() -> dict[str, type[Model]]:
    return {
        normalize_model_name(name): model_class
        for name, model_class in MODEL_REGISTRY.items()
    }


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="YAML model config path for the selected model.",
    )
    parser.add_argument(
        "--embedding-config",
        type=Path,
        help="YAML embedding config path for models that use embeddings.",
    )


def create_model(
    name: str,
    *,
    model_config: str | Path | None = None,
    embedding_config: str | Path | None = None,
) -> Model:
    registry = normalized_registry()
    normalized = normalize_model_name(name)
    if normalized not in registry:
        raise NotImplementedError(f"Unknown model: {name}")
    model_class = registry[normalized]
    return model_class(
        model_config=model_config,
        embedding_config=embedding_config,
    )


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


__all__ = [
    "Action",
    "LinearModel",
    "Model",
    "Page",
    "RandomWalk",
    "SemanticWalk",
    "Title",
    "add_model_args",
    "available_models",
    "create_model",
    *_OPTIONAL_EXPORTS,
]
