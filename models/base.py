from __future__ import annotations

from abc import ABC, abstractmethod

from utils import Action, Page


class Model(ABC):
    """Interface for scoring or generating the next action."""

    def begin_episode(self, start_title: str, target: str) -> None:
        """Reset any per-episode model state before a rollout begins."""

    @abstractmethod
    def sample(self, page: Page, target: str) -> Action | None:
        """Choose one action from the current page toward the target title.

        Args:
            page: The current page and its candidate outgoing links.
            target: The destination title the model is trying to reach.
        """
        raise NotImplementedError
