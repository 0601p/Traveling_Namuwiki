from __future__ import annotations

from pathlib import Path

import torch

from utils import Action, Config, Page

from .networks.neural_target_actor_critic import NeuralTargetActorCritic
from .base import Model


class NeuralTargetA2C(Model):
    """Torch encoder-based actor-critic for target-conditioned navigation."""

    def __init__(
        self,
        *,
        model_config: str | Path | None = None,
        embedding_config: str | Path | None = None,
    ) -> None:
        config = Config(model_config)
        self.checkpoint_path = str(config.value("checkpoint_path"))
        self.device = torch.device(resolve_device(config.value("device")))
        self.network = NeuralTargetActorCritic()
        self.network.to(self.device)
        self.network.eval()
        self._history: set[str] = set()
        self._load_checkpoint()

    # Formerly reset(); use the shared begin_episode hook.
    def begin_episode(self, start_title: str, target: str) -> None:
        del target
        self._history = {start_title}

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

        if self._history:
            penalties = torch.tensor(
                [-0.5 if action in self._history else 0.0 for action in page.actions],
                dtype=torch.float32,
                device=self.device,
            )
            logits = logits + penalties

        action_index = int(torch.argmax(logits).item())
        action = page.actions[action_index]
        self._history.add(action)
        return action

    def _load_checkpoint(self) -> None:
        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.exists():
            return
        payload = torch.load(checkpoint, map_location=self.device)
        config = payload.get("config", {})
        self.network = NeuralTargetActorCritic(**config).to(self.device)
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()


def resolve_device(device: object) -> str:
    raw_device = "auto" if device is None else str(device)
    if raw_device != "auto":
        return raw_device
    return "cuda" if torch.cuda.is_available() else "cpu"
