from __future__ import annotations

from pathlib import Path

from similarity import cached_similarity
from utils import Action, Config, Page

from .base import Model


class LexicalSimilarityGreedy(Model):
    """Greedily choose the outgoing link with highest lexical target similarity."""

    def __init__(
        self,
        *,
        model_config: str | Path | None = None,
        embedding_config: str | Path | None = None,
    ) -> None:
        Config(model_config)
        self._history: set[str] = set()

    def begin_episode(self, start_title: str, target: str) -> None:
        del target
        self._history = {start_title}

    def sample(self, page: Page, target: str) -> Action | None:
        if not page.actions:
            return None
        best_action = max(
            page.actions,
            key=lambda action: (
                cached_similarity(action, target),
                -float(action in self._history),
            ),
        )
        self._history.add(best_action)
        return best_action
