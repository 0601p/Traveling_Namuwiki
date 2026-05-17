from __future__ import annotations

import os
from pathlib import Path

import torch

from .neural_target_a2c import NeuralTargetA2C, NeuralTargetActorCritic


class HindsightTargetA2CV2(NeuralTargetA2C):
    """Encoder actor-critic trained with HER-style hindsight relabeling."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path or os.environ.get(
            "HINDSIGHT_TARGET_A2C_V2_CHECKPOINT"
        ) or str(Path("checkpoints") / "hindsight_target_a2c_v2.pt")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.network = NeuralTargetActorCritic()
        self.network.to(self.device)
        self.network.eval()
        self._history: set[str] = set()
        self._load_checkpoint()
