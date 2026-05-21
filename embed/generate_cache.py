from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

from .embed_model import Embedder


def batched(values: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def write_embedding_cache(
    keys: Sequence[str],
    *,
    output_path: str | Path,
    embedder: Embedder,
    prefix: str | None = None,
    batch_size: int = 64,
) -> int:
    """Write title-keyed embeddings in the JSONL format accepted by CachedEmbedder."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for batch_keys in batched([key.strip() for key in keys if key.strip()], batch_size):
            vectors = embedder.get_embeds(batch_keys, prefix=prefix)
            for key, vector in zip(batch_keys, vectors):
                handle.write(
                    json.dumps(
                        {"title": key, "embedding": vector},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
    return written
