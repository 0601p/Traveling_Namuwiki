from __future__ import annotations

import os
from pathlib import Path

import torch

from utils import Action, Page, cached_similarity

from .hindsight_target_a2c_v2 import HindsightTargetA2CV2


class ResidualHindsightTargetA2CV3(HindsightTargetA2CV2):
    """HER-trained actor with lexical greedy residual prior at inference."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.prior_alpha = float(os.environ.get("RESIDUAL_HER_PRIOR_ALPHA", "5.0"))
        super().__init__(
            checkpoint_path=checkpoint_path
            or os.environ.get("RESIDUAL_HINDSIGHT_TARGET_A2C_V3_CHECKPOINT")
            or str(Path("checkpoints") / "residual_hindsight_target_a2c_v3.pt"),
            device=device,
        )

    @torch.no_grad()
    def sample(self, page: Page, target: str) -> Action | None:
        if not page.actions:
            return None

        texts = [page.title, target, *page.actions]
        embeddings = self.network.encode_texts(texts, device=self.device)
        current_embedding = embeddings[0:1]
        target_embedding = embeddings[1:2]
        action_embeddings = embeddings[2:]
        logits = self.network.actor_logits(
            current_embedding=current_embedding,
            target_embedding=target_embedding,
            action_embeddings=action_embeddings,
        )
        prior = torch.tensor(
            [cached_similarity(action, target) for action in page.actions],
            dtype=torch.float32,
            device=self.device,
        )
        logits = logits + self.prior_alpha * prior

        if self._history:
            penalties = torch.tensor(
                [-0.5 if action in self._history else 0.0 for action in page.actions],
                dtype=torch.float32,
                device=self.device,
            )
            logits = logits + penalties

        action = page.actions[int(torch.argmax(logits).item())]
        self._history.add(action)
        return action
