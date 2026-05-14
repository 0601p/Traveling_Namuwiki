from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from utils import Action, Page

from .base import Model


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(l * r for l, r in zip(left, right))


def as_float_list(values: Iterable[object], *, name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def parse_embedding_record(title: str, values: object) -> tuple[str, list[float]]:
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"Invalid embedding title: {title!r}")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"Embedding for {title!r} must be a sequence of numbers")
    return title.strip(), as_float_list(values, name=f"embedding[{title!r}]")


def load_embeddings(path: str | Path) -> dict[str, list[float]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Embedding file not found: {source}")

    if source.suffix == ".jsonl":
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"Embedding row {line_number} must be an object: {row!r}"
                    )
                rows.append(row)
    else:
        with source.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)

    embeddings: dict[str, list[float]] = {}
    if isinstance(rows, Mapping):
        iterator = rows.items()
    elif isinstance(rows, list):
        iterator = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Embedding row must be an object: {row!r}")
            title = row.get("title")
            values = row.get("embedding", row.get("vector"))
            iterator.append((title, values))
    else:
        raise ValueError("Embedding file must be a JSON object, list, or JSONL records")

    for title, values in iterator:
        key, vector = parse_embedding_record(title, values)
        if key in embeddings and embeddings[key] != vector:
            raise ValueError(f"Duplicate embeddings with different values for {key!r}")
        embeddings[key] = vector

    if not embeddings:
        raise ValueError(f"No embeddings loaded from {source}")

    dimension = len(next(iter(embeddings.values())))
    for title, vector in embeddings.items():
        if len(vector) != dimension:
            raise ValueError(
                f"Inconsistent embedding size for {title!r}: expected {dimension}, got {len(vector)}"
            )
    return embeddings


def load_weights(
    path: str | Path | None,
    *,
    embedding_dim: int,
) -> tuple[list[float], list[float], float]:
    if path is None:
        return [1.0] * embedding_dim, [1.0] * embedding_dim, 0.0

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Weight file not found: {source}")

    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError("Weight file must contain a JSON object")

    if "weights" in payload:
        weights = as_float_list(payload["weights"], name="weights")
        if len(weights) != embedding_dim * 2:
            raise ValueError(
                f"'weights' must have size {embedding_dim * 2}, got {len(weights)}"
            )
        link_weights = weights[:embedding_dim]
        target_weights = weights[embedding_dim:]
    else:
        link_weights = as_float_list(
            payload.get("link_weights", [1.0] * embedding_dim),
            name="link_weights",
        )
        target_weights = as_float_list(
            payload.get("target_weights", [1.0] * embedding_dim),
            name="target_weights",
        )
        if len(link_weights) != embedding_dim:
            raise ValueError(
                f"link_weights must have size {embedding_dim}, got {len(link_weights)}"
            )
        if len(target_weights) != embedding_dim:
            raise ValueError(
                f"target_weights must have size {embedding_dim}, got {len(target_weights)}"
            )

    bias = float(payload.get("bias", 0.0))
    return link_weights, target_weights, bias


class LinearModel(Model):
    """Score candidate links with a linear model over link and target embeddings."""

    def __init__(
        self,
        *,
        embeddings_path: str,
        weights_path: str | None = None,
    ) -> None:
        self.embeddings = load_embeddings(embeddings_path)
        self.embedding_dim = len(next(iter(self.embeddings.values())))
        self.link_weights, self.target_weights, self.bias = load_weights(
            weights_path,
            embedding_dim=self.embedding_dim,
        )

    def score(self, action: Action, target: str) -> float | None:
        action_embedding = self.embeddings.get(action)
        target_embedding = self.embeddings.get(target)
        if action_embedding is None or target_embedding is None:
            return None
        return (
            dot(self.link_weights, action_embedding)
            + dot(self.target_weights, target_embedding)
            + self.bias
        )

    def sample(self, page: Page, target: str) -> Action | None:
        best_action = None
        best_score = None

        for action in page.actions:
            score = self.score(action, target)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_action = action
                best_score = score

        return best_action
