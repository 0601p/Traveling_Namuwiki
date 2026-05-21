from __future__ import annotations

import random
from pathlib import Path

from utils import Action, Config, Page

from .base import Model


class RandomWalk(Model):
    """Uniformly sample one outgoing action."""

    def __init__(
        self,
        *,
        model_config: str | Path | None = None,
        embedding_config: str | Path | None = None,
    ) -> None:
        config = Config(model_config)
        seed = config.value("seed")
        self.rng = random.Random(int(seed) if seed is not None else None)

    def sample(self, page: Page, target: str) -> Action | None:
        """Ignore the target input and pick a random outgoing link."""
        if not page.actions:
            return None
        return self.rng.choice(list(page.actions))
