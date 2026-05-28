from __future__ import annotations

from pathlib import Path

from embed import CachedEmbedder, EmbeddingModel
from similarity import get_vector_metric
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
        config = Config(model_config)
        self.avoid_visited = bool(config.value("avoid_visited"))
        self.metric = get_vector_metric(str(config.value("metric")))
        embed_config = Config(embedding_config)
        self.embedder = EmbeddingModel(
            config=embed_config.data,
        )
        self.query_prefix = None if isinstance(self.embedder.backend, CachedEmbedder) else self.QUERY_PREFIX
        self.passage_prefix = None if isinstance(self.embedder.backend, CachedEmbedder) else self.PASSAGE_PREFIX
        self._history: set[str] = set()

    def begin_episode(self, start_title: str, target: str) -> None:
        del target
        self._history = {start_title}

    def sample(self, page: Page, target: str) -> Action | None:
        actions = list(page.actions)
        if not actions:
            return None

        if self.avoid_visited:
            unvisited = [action for action in actions if action not in self._history]
            candidates = unvisited or actions
        else:
            candidates = actions

        target_vector = self.embedder.get_embed(target, prefix=self.query_prefix)
        candidate_vectors = self.embedder.get_embeds(candidates, prefix=self.passage_prefix)
        scores = [self.metric(candidate_vector, target_vector) for candidate_vector in candidate_vectors]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        chosen = candidates[best_index]
        if self.avoid_visited:
            self._history.add(chosen)
        return chosen
