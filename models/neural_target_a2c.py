from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from utils import Action, Page

from .base import Model


TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def ngram_ids(text: str, *, bucket_size: int, min_n: int = 2, max_n: int = 4) -> list[int]:
    compact = normalize_text(text).replace(" ", "")
    if not compact:
        return [0]

    ids = [1]
    for n in range(min_n, max_n + 1):
        if len(compact) < n:
            ids.append(stable_hash(compact, bucket_size))
            continue
        for index in range(len(compact) - n + 1):
            ids.append(stable_hash(compact[index : index + n], bucket_size))
    return ids[:128]


def stable_hash(text: str, bucket_size: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return 2 + int.from_bytes(digest, "little") % (bucket_size - 2)


class NeuralTargetActorCritic(nn.Module):
    def __init__(
        self,
        *,
        bucket_size: int = 50000,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.bucket_size = bucket_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.EmbeddingBag(
            bucket_size,
            embedding_dim,
            mode="mean",
            padding_idx=0,
        )
        pair_dim = embedding_dim * 4
        action_dim = embedding_dim * 7
        self.actor = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_texts(self, texts: Sequence[str], *, device: torch.device) -> torch.Tensor:
        ids: list[int] = []
        offsets: list[int] = []
        for text in texts:
            offsets.append(len(ids))
            ids.extend(ngram_ids(text, bucket_size=self.bucket_size))
        id_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        offset_tensor = torch.tensor(offsets, dtype=torch.long, device=device)
        return self.embedding(id_tensor, offset_tensor)

    def actor_logits(
        self,
        *,
        current_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        action_count = action_embeddings.shape[0]
        current = current_embedding.expand(action_count, -1)
        target = target_embedding.expand(action_count, -1)
        features = torch.cat(
            [
                current,
                target,
                action_embeddings,
                action_embeddings * target,
                torch.abs(action_embeddings - target),
                action_embeddings * current,
                torch.abs(action_embeddings - current),
            ],
            dim=1,
        )
        return self.actor(features).squeeze(-1)

    def value(self, current_embedding: torch.Tensor, target_embedding: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [
                current_embedding,
                target_embedding,
                current_embedding * target_embedding,
                torch.abs(current_embedding - target_embedding),
            ],
            dim=-1,
        )
        return self.critic(features).squeeze(-1)


class NeuralTargetA2C(Model):
    """Torch encoder-based actor-critic for target-conditioned navigation."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path or os.environ.get(
            "NEURAL_TARGET_A2C_CHECKPOINT"
        ) or str(
            Path("checkpoints") / "neural_target_a2c.pt"
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.network = NeuralTargetActorCritic()
        self.network.to(self.device)
        self.network.eval()
        self._history: set[str] = set()
        self._load_checkpoint()

    def reset(self, *, start_title: str, target_title: str) -> None:
        del target_title
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


def save_checkpoint(
    *,
    path: Path,
    network: NeuralTargetActorCritic,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": network.state_dict(),
            "config": {
                "bucket_size": network.bucket_size,
                "embedding_dim": network.embedding_dim,
                "hidden_dim": network.hidden_dim,
            },
            "metrics": metrics,
        },
        path,
    )


def checkpoint_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return None
