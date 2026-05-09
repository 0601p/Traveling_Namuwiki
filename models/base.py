from __future__ import annotations

from abc import ABC, abstractmethod

from utils import Action, Page

class Model(ABC):
    """Interface for scoring or generating the next action."""

    @abstractmethod
    def sample(self, page: Page, target: str, history: list[str]) -> Action | None:
        """Choose one action from the current page toward the target title.

        Args:
            page: The current page and its candidate outgoing links.
            target: The destination title the model is trying to reach.
            history: The titles visited so far in the current rollout.
        """
        raise NotImplementedError
