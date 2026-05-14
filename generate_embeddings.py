from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from utils import ACTIONS_DATASET, RAW_DATASET, Title


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate title-keyed embeddings for Traveling Namuwiki."
    )
    parser.add_argument(
        "--actions-path",
        default=ACTIONS_DATASET,
        help="Dataset id or local dataset path for graph actions.",
    )
    parser.add_argument(
        "--raw-path",
        default=RAW_DATASET,
        help="Dataset id or local dataset path for raw documents.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split used for actions and raw text lookup.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Hugging Face encoder used to create embeddings.",
    )
    parser.add_argument(
        "--text-source",
        choices=["title", "raw", "raw_or_title"],
        default="raw_or_title",
        help="Text source for each title before embedding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoder inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer max length.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device, e.g. cpu, cuda, cuda:0. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output JSONL path for title-keyed embedding vectors.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print a progress log every N embedding batches.",
    )
    return parser.parse_args()


def resolve_device(raw_device: str) -> str:
    if raw_device != "auto":
        return raw_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def iter_titles(actions_path: str, split: str) -> Iterable[Title]:
    seen: set[Title] = set()
    for row in load_dataset(actions_path, split=split):
        title = str(row.get("title") or "").strip()
        if title and title not in seen:
            seen.add(title)
            yield title

        for action in row.get("actions", []):
            candidate = str(action or "").strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                yield candidate


def load_raw_texts(raw_path: str, split: str, allowed_titles: set[Title]) -> dict[Title, str]:
    raws: dict[Title, str] = {}
    for row in load_dataset(raw_path, split=split):
        title = str(row.get("title") or "").strip()
        if title in allowed_titles and title not in raws:
            raws[title] = str(row.get("text") or "")
    return raws


def select_text(
    title: Title,
    *,
    raws: dict[Title, str],
    text_source: str,
) -> str:
    if text_source == "title":
        return title
    if text_source == "raw":
        if title not in raws:
            raise KeyError(f"Missing raw text for title {title!r}")
        return raws[title]
    return raws.get(title, title)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked = last_hidden_state * mask
    counts = mask.sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / counts


def encode_texts(
    texts: list[str],
    *,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_length: int,
) -> list[list[float]]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        normalized = F.normalize(pooled, p=2, dim=1)
    return normalized.cpu().tolist()


def batched(values: list[Title], batch_size: int) -> Iterable[list[Title]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    log(
        "[generate_embeddings] "
        f"device={device} torch_cuda_available={torch.cuda.is_available()} "
        f"model_name={args.model_name} text_source={args.text_source}"
    )

    log("[generate_embeddings] collecting titles from actions dataset")
    titles = list(iter_titles(args.actions_path, args.split))
    log(f"[generate_embeddings] collected {len(titles)} unique titles")

    raws = (
        load_raw_texts(args.raw_path, args.split, set(titles))
        if args.text_source != "title"
        else {}
    )
    if args.text_source != "title":
        log(f"[generate_embeddings] loaded raw text for {len(raws)} titles")

    log("[generate_embeddings] loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    log("[generate_embeddings] loading encoder model")
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    log("[generate_embeddings] model ready")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    written_titles = 0
    total_batches = (len(titles) + args.batch_size - 1) // args.batch_size
    log(f"[generate_embeddings] streaming embeddings to {args.output_path}")
    with args.output_path.open("w", encoding="utf-8") as handle:
        for batch_index, batch_titles in enumerate(batched(titles, args.batch_size), start=1):
            batch_texts = [
                select_text(
                    title,
                    raws=raws,
                    text_source=args.text_source,
                )
                for title in batch_titles
            ]
            batch_vectors = encode_texts(
                batch_texts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=args.max_length,
            )
            for title, vector in zip(batch_titles, batch_vectors):
                handle.write(
                    json.dumps(
                        {"title": title, "embedding": vector},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written_titles += 1
            handle.flush()
            if (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % max(args.log_every, 1) == 0
            ):
                log(
                    "[generate_embeddings] "
                    f"batch {batch_index}/{total_batches} "
                    f"embedded_titles={written_titles}"
                )

    log("[generate_embeddings] completed successfully")
    print(
        json.dumps(
            {
                "output_path": str(args.output_path),
                "titles": written_titles,
                "model_name": args.model_name,
                "text_source": args.text_source,
                "device": device,
                "embedding_dim": len(batch_vectors[0]) if written_titles else 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
