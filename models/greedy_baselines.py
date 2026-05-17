from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from utils import Action, DEFAULT_LM_BACKBONE, Page, cached_similarity

from .base import Model


class LexicalSimilarityGreedy(Model):
    """Greedily choose the outgoing link with highest lexical target similarity."""

    def __init__(self) -> None:
        self._history: set[str] = set()

    def reset(self, *, start_title: str, target_title: str) -> None:
        del target_title
        self._history = {start_title}

    def sample(self, page: Page, target: str) -> Action | None:
        if not page.actions:
            return None
        best_action = max(
            page.actions,
            key=lambda action: (
                cached_similarity(action, target),
                -float(action in self._history),
            ),
        )
        self._history.add(best_action)
        return best_action


class LmEmbeddingGreedy(Model):
    """Greedily choose the outgoing link with highest LM cosine similarity to target."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int = 32,
    ) -> None:
        self.model_name = model_name or os.environ.get(
            "LM_GREEDY_BACKBONE",
            DEFAULT_LM_BACKBONE,
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_length = int(os.environ.get("LM_GREEDY_MAX_LENGTH", max_length))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,
        )
        self.encoder = AutoModel.from_pretrained(
            self.model_name,
            local_files_only=True,
        ).to(self.device)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self._cache: dict[str, torch.Tensor] = {}
        self._history: set[str] = set()

    def reset(self, *, start_title: str, target_title: str) -> None:
        del target_title
        self._history = {start_title}

    @torch.no_grad()
    def sample(self, page: Page, target: str) -> Action | None:
        if not page.actions:
            return None
        embeddings = self._encode([target, *page.actions])
        target_embedding = F.normalize(embeddings[0:1], dim=1)
        action_embeddings = F.normalize(embeddings[1:], dim=1)
        scores = torch.matmul(action_embeddings, target_embedding.T).squeeze(1)
        if self._history:
            penalties = torch.tensor(
                [-0.05 if action in self._history else 0.0 for action in page.actions],
                dtype=torch.float32,
                device=self.device,
            )
            scores = scores + penalties
        action = page.actions[int(torch.argmax(scores).item())]
        self._history.add(action)
        return action

    @torch.no_grad()
    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            batch = self.tokenizer(
                missing,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(outputs.dtype)
            embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            for text, embedding in zip(missing, embeddings.detach().cpu()):
                self._cache[text] = embedding
        return torch.stack([self._cache[text] for text in texts]).to(self.device)
