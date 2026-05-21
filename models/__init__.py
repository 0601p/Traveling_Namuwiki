from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

from utils import Action, Page, Title

from .base import Model


MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "ar_walk": (".ar_walk", "AutoregressiveWalk"),
    "hindsighttargeta2cv2": (".hindsight_target_a2c_v2", "HindsightTargetA2CV2"),
    "lexicalsimilaritygreedy": (".greedy_baselines", "LexicalSimilarityGreedy"),
    "linear": (".linear", "LinearModel"),
    "neuraltargeta2c": (".neural_target_a2c", "NeuralTargetA2C"),
    "randomwalk": (".randomwalk", "RandomWalk"),
    "residualhindsighttargeta2cv3": (
        ".residual_hindsight_target_a2c_v3",
        "ResidualHindsightTargetA2CV3",
    ),
    "semanticwalk": (".semantic_walk", "SemanticWalk"),
}


def normalize_model_name(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "")


def normalized_registry() -> dict[str, tuple[str, str]]:
    return {
        normalize_model_name(name): model_spec
        for name, model_spec in MODEL_REGISTRY.items()
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
    module_name, class_name = registry[normalized]
    module = import_module(module_name, package=__name__)
    model_cls = getattr(module, class_name)
    return model_cls(
        model_config=model_config,
        embedding_config=embedding_config,
    )


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


__all__ = [
    "Action",
    "Model",
    "Page",
    "Title",
    "add_model_args",
    "available_models",
    "create_model",
]
