from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from embed import EmbeddingModel, as_float_list
from utils import Action, Config, Page

from .base import Model


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(l * r for l, r in zip(left, right))


def load_weights(
    path: str | Path | None,
    *,
    embedding_dim: int,
) -> tuple[list[float], list[float], list[float], float]:
    if path is None:
        return (
            [0.0] * embedding_dim,
            [0.0] * embedding_dim,
            [1.0] * embedding_dim,
            0.0,
        )

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Weight file not found: {source}")

    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError("Weight file must contain a JSON object")

    if "weights" in payload:
        weights = as_float_list(payload["weights"], name="weights")
        if len(weights) == embedding_dim * 2:
            link_weights = weights[:embedding_dim]
            target_weights = weights[embedding_dim:]
            interaction_weights = [0.0] * embedding_dim
        elif len(weights) == embedding_dim * 3:
            link_weights = weights[:embedding_dim]
            target_weights = weights[embedding_dim : embedding_dim * 2]
            interaction_weights = weights[embedding_dim * 2 :]
        else:
            raise ValueError(
                f"'weights' must have size {embedding_dim * 2} or {embedding_dim * 3}, got {len(weights)}"
            )
    else:
        link_weights = as_float_list(
            payload.get("link_weights", [0.0] * embedding_dim),
            name="link_weights",
        )
        target_weights = as_float_list(
            payload.get("target_weights", [0.0] * embedding_dim),
            name="target_weights",
        )
        interaction_weights = as_float_list(
            payload.get("interaction_weights", [1.0] * embedding_dim),
            name="interaction_weights",
        )
        if len(link_weights) != embedding_dim:
            raise ValueError(
                f"link_weights must have size {embedding_dim}, got {len(link_weights)}"
            )
        if len(target_weights) != embedding_dim:
            raise ValueError(
                f"target_weights must have size {embedding_dim}, got {len(target_weights)}"
            )
        if len(interaction_weights) != embedding_dim:
            raise ValueError(
                f"interaction_weights must have size {embedding_dim}, got {len(interaction_weights)}"
            )

    bias = float(payload.get("bias", 0.0))
    return link_weights, target_weights, interaction_weights, bias


class LinearModel(Model):
    """Score candidate links with a linear model over link, target, and interaction features."""

    def __init__(
        self,
        *,
        model_config: str | Path | None = None,
        embedding_config: str | Path | None = None,
    ) -> None:
        config = Config(model_config)
        embed_config = Config(embedding_config)
        weights_path = config.value("weights_path")
        self.embedder = EmbeddingModel(config=embed_config.data)
        self.embedding_dim = self.embedder.get_embed_dim()
        (
            self.link_weights,
            self.target_weights,
            self.interaction_weights,
            self.bias,
        ) = load_weights(
            weights_path,
            embedding_dim=self.embedding_dim,
        )

    def score(self, action: Action, target: str) -> float | None:
        action_embedding = self.embedder.get_embed(action)
        target_embedding = self.embedder.get_embed(target)
        interaction_embedding = [
            link_value * target_value
            for link_value, target_value in zip(action_embedding, target_embedding)
        ]
        return (
            dot(self.link_weights, action_embedding)
            + dot(self.target_weights, target_embedding)
            + dot(self.interaction_weights, interaction_embedding)
            + self.bias
        )

    def sample(self, page: Page, target: str) -> Action | None:
        best_action = None
        best_score = None

        for action in page.actions:
            score = self.score(action, target)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_action = action
                best_score = score

        return best_action
