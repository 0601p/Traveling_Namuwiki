from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_rows(dataset_path: str, *, split: str = "train") -> Iterable[dict]:
    """Load rows from a local JSON/JSONL file or a Hugging Face dataset id."""
    path = Path(dataset_path)
    if path.exists():
        yield from _load_local_rows(path, split=split)
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for remote dataset ids. "
            "Install it with `pip install datasets` or pass a local dataset path."
        ) from exc

    try:
        yield from load_dataset(dataset_path, split=split)
        return
    except Exception:
        cached_path = _find_hf_cached_dataset_dir(dataset_path)
        if cached_path is None:
            raise
        yield from _load_local_rows(cached_path, split=split)


def _load_local_rows(path: Path, *, split: str) -> Iterable[dict]:
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            yield from _read_jsonl(path)
            return
        if suffix == ".json":
            yield from _read_json(path, split=split)
            return
        if suffix == ".arrow":
            yield from _read_arrow_files([path])
            return
        raise ValueError(f"Unsupported local dataset file: {path}")

    candidates = [
        path / f"{split}.jsonl",
        path / f"{split}.json",
        path / f"{split}.ndjson",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix == ".json":
                yield from _read_json(candidate, split=split)
            else:
                yield from _read_jsonl(candidate)
            return

    arrow_files = sorted(path.glob(f"*-{split}.arrow"))
    if not arrow_files:
        arrow_files = sorted(path.glob(f"*-{split}-*.arrow"))
    if arrow_files:
        yield from _read_arrow_files(arrow_files)
        return

    raise FileNotFoundError(
        f"Could not find a local split file under {path} for split={split!r}."
    )


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_json(path: Path, *, split: str) -> Iterable[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for row in payload:
            yield row
        return

    if isinstance(payload, dict):
        rows = payload.get(split)
        if isinstance(rows, list):
            for row in rows:
                yield row
            return

    raise ValueError(f"Unsupported JSON dataset structure in {path}")


def _read_arrow_files(paths: list[Path]) -> Iterable[dict]:
    try:
        from datasets import Dataset, concatenate_datasets
    except ImportError as exc:
        raise ImportError(
            "datasets is required for local Arrow datasets. "
            "Install it with `pip install datasets`."
        ) from exc

    datasets_list = [Dataset.from_file(str(path)) for path in paths]
    if len(datasets_list) == 1:
        dataset = datasets_list[0]
    else:
        dataset = concatenate_datasets(datasets_list)
    for row in dataset:
        yield row


def _find_hf_cached_dataset_dir(dataset_path: str) -> Path | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
    dataset_dir = cache_root / dataset_path.replace("/", "___").lower()
    if not dataset_dir.exists():
        return None

    version_root = dataset_dir / "default" / "0.0.0"
    if not version_root.exists():
        return None

    candidates = sorted(path for path in version_root.iterdir() if path.is_dir())
    if not candidates:
        return None
    return candidates[-1]
