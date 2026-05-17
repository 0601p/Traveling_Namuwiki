from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from models.neural_target_a2c import ngram_ids


class BeamHerReranker(nn.Module):
    def __init__(
        self,
        *,
        bucket_size: int = 50000,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        feature_dim: int = 6,
    ) -> None:
        super().__init__()
        self.bucket_size = bucket_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.embedding = nn.EmbeddingBag(
            bucket_size,
            embedding_dim,
            mode="mean",
            padding_idx=0,
        )
        input_dim = embedding_dim * 7 + feature_dim
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
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

    def scores(
        self,
        *,
        current: str,
        target: str,
        actions: Sequence[str],
        scalar_features: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        embeddings = self.encode_texts([current, target, *actions], device=device)
        current_embedding = embeddings[0:1]
        target_embedding = embeddings[1:2]
        action_embeddings = embeddings[2:]
        action_count = action_embeddings.shape[0]
        current_batch = current_embedding.expand(action_count, -1)
        target_batch = target_embedding.expand(action_count, -1)
        features = torch.cat(
            [
                current_batch,
                target_batch,
                action_embeddings,
                action_embeddings * target_batch,
                torch.abs(action_embeddings - target_batch),
                action_embeddings * current_batch,
                torch.abs(action_embeddings - current_batch),
                scalar_features,
            ],
            dim=1,
        )
        return self.scorer(features).squeeze(-1)


def save_beam_reranker_checkpoint(
    *,
    path: Path,
    model: BeamHerReranker,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "bucket_size": model.bucket_size,
                "embedding_dim": model.embedding_dim,
                "hidden_dim": model.hidden_dim,
                "feature_dim": model.feature_dim,
            },
            "metrics": metrics,
        },
        path,
    )


def load_beam_reranker_checkpoint(
    *,
    path: Path,
    device: torch.device,
) -> BeamHerReranker:
    payload = torch.load(path, map_location=device)
    model = BeamHerReranker(**payload.get("config", {})).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
