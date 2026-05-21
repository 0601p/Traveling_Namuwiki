from __future__ import annotations

from pathlib import Path

import torch

from similarity import cached_similarity
from utils import Action, Config, Page

from .hindsight_target_a2c_v2 import HindsightTargetA2CV2


class ResidualHindsightTargetA2CV3(HindsightTargetA2CV2):
    """HER-trained actor with lexical greedy residual prior at inference."""

    def __init__(
        self,
        *,
        model_config: str | Path | None = None,
        embedding_config: str | Path | None = None,
    ) -> None:
        config = Config(model_config)
        self.prior_alpha = float(config.value("prior_alpha"))
        super().__init__(
            model_config=model_config,
            embedding_config=embedding_config,
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
