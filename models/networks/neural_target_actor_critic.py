from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

import torch
from torch import nn


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
