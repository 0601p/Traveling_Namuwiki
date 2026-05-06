from __future__ import annotations

import random

from utils import Action, Page

from .base import Model


class RandomWalk(Model):
    """Uniformly sample one outgoing action."""

    def __init__(self, seed: int | None = 0) -> None:
        self.rng = random.Random(seed)

    def sample(self, page: Page, target: str) -> Action | None:
        if not page.actions:
            return None
        return self.rng.choice(list(page.actions))
