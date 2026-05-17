from __future__ import annotations

from importlib import import_module

from utils import Action, Page, Title

from .base import Model


MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "hindsighttargeta2cv2": (".hindsight_target_a2c_v2", "HindsightTargetA2CV2"),
    "lexicalsimilaritygreedy": (".greedy_baselines", "LexicalSimilarityGreedy"),
    "lmembeddinggreedy": (".greedy_baselines", "LmEmbeddingGreedy"),
    "neuraltargeta2c": (".neural_target_a2c", "NeuralTargetA2C"),
    "randomwalk": (".randomwalk", "RandomWalk"),
    "residualhindsighttargeta2cv3": (
        ".residual_hindsight_target_a2c_v3",
        "ResidualHindsightTargetA2CV3",
    ),
}


def create_model(name: str) -> Model:
    normalized = name.strip().lower().replace("_", "").replace("-", "")
    if normalized not in MODEL_REGISTRY:
        raise NotImplementedError(f"Unknown model: {name}")
    module_name, class_name = MODEL_REGISTRY[normalized]
    module = import_module(module_name, package=__name__)
    model_cls = getattr(module, class_name)
    return model_cls()


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


__all__ = [
    "Action",
    "Model",
    "Page",
    "Title",
    "available_models",
    "create_model",
]
