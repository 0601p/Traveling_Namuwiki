from __future__ import annotations

import json
import os
from typing import Sequence

import torch

import train_hindsight_target_a2c_v2 as base
from models.neural_target_a2c import NeuralTargetActorCritic
from utils import cached_similarity


PRIOR_ALPHA = float(os.environ.get("RESIDUAL_HER_PRIOR_ALPHA", "5.0"))


def residual_forward_state(
    *,
    network: NeuralTargetActorCritic,
    current: str,
    target: str,
    actions: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, value = original_forward_state(
        network=network,
        current=current,
        target=target,
        actions=actions,
        device=device,
    )
    prior = torch.tensor(
        [cached_similarity(action, target) for action in actions],
        dtype=torch.float32,
        device=device,
    )
    return logits + PRIOR_ALPHA * prior, value


original_forward_state = base.forward_state
base.forward_state = residual_forward_state


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "wrapper": "residual_hindsight_target_a2c_v3",
                "prior_alpha": PRIOR_ALPHA,
            }
        ),
        flush=True,
    )
    base.main()
