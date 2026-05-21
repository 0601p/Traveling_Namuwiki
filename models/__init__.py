from __future__ import annotations

import argparse
import inspect

from embed import DEFAULT_ON_THE_FLY_CONFIG
from .config import DEFAULT_LINEAR_CONFIG
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


def normalize_model_name(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "")


def normalized_registry() -> dict[str, type[Model]]:
    return {
        normalize_model_name(name): model_class
        for name, model_class in MODEL_REGISTRY.items()
    }


def add_linear_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-config",
        default=str(DEFAULT_LINEAR_CONFIG),
        help="YAML model config path, e.g. linear weights_path.",
    )


def add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-config",
        default=str(DEFAULT_ON_THE_FLY_CONFIG),
        help="YAML embedding config path for models with on-the-fly embeddings.",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    add_linear_args(parser)
    add_embedding_args(parser)


def accepted_kwargs(
    model_class: type[Model],
    kwargs: dict[str, object],
) -> dict[str, object]:
    signature = inspect.signature(model_class)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs
    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }


def create_model(name: str, **kwargs: object) -> Model:
    registry = normalized_registry()
    normalized = normalize_model_name(name)
    if normalized not in registry:
        raise NotImplementedError(f"Unknown model: {name}")
    model_class = registry[normalized]
    return model_class(**accepted_kwargs(model_class, kwargs))


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
]
