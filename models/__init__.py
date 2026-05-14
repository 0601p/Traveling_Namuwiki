from __future__ import annotations

import argparse

from utils import Action, Page, Title

from .base import Model
from .linear import LinearModel
from .randomwalk import RandomWalk


MODEL_REGISTRY: dict[str, type[Model]] = {
    "randomwalk": RandomWalk,
    "linear": LinearModel,
}


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embeddings-path",
        help="JSON/JSONL file mapping titles to embedding vectors. Required for --model linear.",
    )
    parser.add_argument(
        "--weights-path",
        help="Optional JSON file with linear weights and bias for --model linear.",
    )


def create_model(name: str, **kwargs: object) -> Model:
    normalized = name.strip().lower().replace("_", "").replace("-", "")
    if normalized not in MODEL_REGISTRY:
        raise NotImplementedError(f"Unknown model: {name}")
    if normalized == "linear":
        embeddings_path = kwargs.get("embeddings_path")
        weights_path = kwargs.get("weights_path")
        if not embeddings_path:
            raise ValueError("--embeddings-path is required for --model linear")
        return LinearModel(
            embeddings_path=str(embeddings_path),
            weights_path=str(weights_path) if weights_path else None,
        )
    return MODEL_REGISTRY[normalized]()


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


__all__ = [
    "Action",
    "LinearModel",
    "Model",
    "Page",
    "RandomWalk",
    "Title",
    "add_model_args",
    "available_models",
    "create_model",
]
