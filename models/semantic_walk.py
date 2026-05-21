from __future__ import annotations

from pathlib import Path

from embed import DEFAULT_ON_THE_FLY_CONFIG, EmbeddingModel
from utils import Action, Page

from .base import Model


class SemanticWalk(Model):
    """Choose the outgoing action with the highest semantic similarity to target.

    This is a semantic walking baseline using multilingual E5 embeddings.
    The model is inference-only and does not perform any training.
    """

    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        *,
        embedding_config: str | Path | None = DEFAULT_ON_THE_FLY_CONFIG,
    ) -> None:
        self.embedder = EmbeddingModel(
            config_path=embedding_config,
        )

    def _score(self, action_embedding: list[float], target_embedding: list[float]) -> float:
        return sum(
            action_value * target_value
            for action_value, target_value in zip(action_embedding, target_embedding)
        )

    def sample(self, page: Page, target: str) -> Action | None:
        actions = list(page.actions)
        if not actions:
            return None

        target_vector = self.embedder.get_embed(target, prefix=self.QUERY_PREFIX)
        action_vectors = self.embedder.get_embeds(actions, prefix=self.PASSAGE_PREFIX)
        scores = [
            self._score(action_vector, target_vector)
            for action_vector in action_vectors
        ]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        return actions[best_index]
