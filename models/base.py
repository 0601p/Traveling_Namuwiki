from __future__ import annotations

from abc import ABC, abstractmethod

from utils import Action, Page


class Model(ABC):
    """Interface for scoring or generating the next action."""

    def reset(self, *, start_title: str, target_title: str) -> None:
        """Reset episode-local state before a new walk starts."""
        del start_title, target_title

    @abstractmethod
    def sample(self, page: Page, target: str) -> Action | None:
        """Choose one action from the current page toward the target title."""
        raise NotImplementedError
