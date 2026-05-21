from __future__ import annotations

from pathlib import Path

import torch

from utils import Config

from .networks.neural_target_actor_critic import NeuralTargetActorCritic
from .neural_target_a2c import NeuralTargetA2C


class HindsightTargetA2CV2(NeuralTargetA2C):
    """Encoder actor-critic trained with HER-style hindsight relabeling."""

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


def resolve_device(device: object) -> str:
    raw_device = "auto" if device is None else str(device)
    if raw_device != "auto":
        return raw_device
    return "cuda" if torch.cuda.is_available() else "cpu"
