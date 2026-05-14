from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from utils import Action, Page

from .base import Model

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class GreedyEmbeddingSimilarity(Model):
    """Choose the outgoing action with the highest semantic similarity to target.

    This is a greedy semantic-search baseline using multilingual E5 embeddings.
    The model is inference-only and does not perform any training.
    """

    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "GreedyEmbeddingSimilarity requires sentence-transformers. "
                "Install it with: pip install sentence-transformers torch numpy"
            ) from exc

        self.encoder: SentenceTransformer = SentenceTransformer(
            model_name, device=device
        )
        self.batch_size = batch_size
        self._embed_cache: dict[str, np.ndarray] = {}

    def _encode(self, prefixed_texts: list[str]) -> np.ndarray:
        """Encode only uncached texts in a batch, then return vectors in order."""
        missing = [text for text in prefixed_texts if text not in self._embed_cache]
        if missing:
            vectors = self.encoder.encode(
                missing,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vector in zip(missing, vectors):
                self._embed_cache[text] = vector
        return np.stack([self._embed_cache[text] for text in prefixed_texts])

    def sample(self, page: Page, target: str) -> Action | None:
        actions = list(page.actions)
        if not actions:
            return None

        target_text = self.QUERY_PREFIX + target
        action_texts = [self.PASSAGE_PREFIX + action for action in actions]

        target_vector = self._encode([target_text])[0]
        action_vectors = self._encode(action_texts)

        scores = action_vectors @ target_vector
        best_index = int(np.argmax(scores))
        return actions[best_index]
