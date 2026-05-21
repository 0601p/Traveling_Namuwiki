from __future__ import annotations

from pathlib import Path

from embed import EmbeddingModel
from utils import Action, Config, Page

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
        model_config: str | Path | None = None,
        embedding_config: str | Path | None = None,
    ) -> None:
        Config(model_config)
        embed_config = Config(embedding_config)
        self.embedder = EmbeddingModel(
            config=embed_config.data,
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
