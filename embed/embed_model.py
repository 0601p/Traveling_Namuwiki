from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


Embedding = list[float]
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
TEXT_SOURCES = {"title", "raw", "raw_or_title"}
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "embed"
DEFAULT_ON_THE_FLY_CONFIG = CONFIG_DIR / "on-the-fly.yaml"
DEFAULT_FROM_CACHE_CONFIG = CONFIG_DIR / "from-cache.yaml"


class Encoder(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        ...


class Embedder(ABC):
    """Common interface for cached and on-the-fly embedding providers."""

    def get_embed(self, text: str, *, prefix: str | None = None) -> Embedding:
        return self.get_embeds([text], prefix=prefix)[0]

    @abstractmethod
    def get_embed_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_embeds(
        self,
        texts: Sequence[str],
        *,
        prefix: str | None = None,
    ) -> list[Embedding]:
        raise NotImplementedError

    def _normalize_key(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Embedding text must be a non-empty string: {text!r}")
        return text.strip()


def as_float_list(values: Iterable[object], *, name: str) -> Embedding:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def parse_embedding_record(text: object, values: object) -> tuple[str, Embedding]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Invalid embedding key: {text!r}")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"Embedding for {text!r} must be a sequence of numbers")
    return text.strip(), as_float_list(values, name=f"embedding[{text!r}]")


def load_embeddings(path: str | Path) -> dict[str, Embedding]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Embedding file not found: {source}")

    if source.suffix == ".jsonl":
        rows: object = []
        with source.open("r", encoding="utf-8-sig") as handle:
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
        with source.open("r", encoding="utf-8-sig") as handle:
            rows = json.load(handle)

    embeddings: dict[str, Embedding] = {}
    if isinstance(rows, Mapping):
        iterator = rows.items()
    elif isinstance(rows, list):
        iterator = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"Embedding row must be an object: {row!r}")
            text = row.get("title", row.get("text", row.get("key")))
            values = row.get("embedding", row.get("vector"))
            iterator.append((text, values))
    else:
        raise ValueError("Embedding file must be a JSON object, list, or JSONL records")

    for text, values in iterator:
        key, vector = parse_embedding_record(text, values)
        if key in embeddings and embeddings[key] != vector:
            raise ValueError(f"Duplicate embeddings with different values for {key!r}")
        embeddings[key] = vector

    if not embeddings:
        raise ValueError(f"No embeddings loaded from {source}")

    dimension = len(next(iter(embeddings.values())))
    for key, vector in embeddings.items():
        if len(vector) != dimension:
            raise ValueError(
                f"Inconsistent embedding size for {key!r}: expected {dimension}, got {len(vector)}"
            )
    return embeddings


def parse_yaml_scalar(value: str) -> object:
    raw = value.strip()
    lowered = raw.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        return raw


def load_embedding_config(path: str | Path) -> dict[str, object]:
    """Load a flat key/value YAML config without requiring PyYAML."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Embedding config not found: {source}")

    config: dict[str, object] = {}
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Invalid YAML line {line_number} in {source}: {line!r}")
        key, value = stripped.split(":", 1)
        key = key.strip().lstrip("\ufeff")
        if not key:
            raise ValueError(f"Missing YAML key on line {line_number} in {source}")
        config[key] = parse_yaml_scalar(value)
    return config


def require_config_value(
    config: Mapping[str, object],
    key: str,
    value: object,
) -> object:
    if value is None:
        raise ValueError(f"Missing required embedding config value: {key}")
    return value


class CachedEmbedder(Embedder):
    """Read embeddings from a fixed cache and fail on missing keys."""

    def __init__(
        self,
        *,
        embeddings_path: str | Path | None = None,
        embeddings: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        if embeddings_path is not None and embeddings is not None:
            raise ValueError("Pass either embeddings_path or embeddings, not both")
        if embeddings_path is None and embeddings is None:
            raise ValueError("CachedEmbedder requires embeddings_path or embeddings")

        self.embeddings: dict[str, Embedding] = {}
        if embeddings_path is not None:
            self.embeddings.update(load_embeddings(embeddings_path))
        elif embeddings is not None:
            for key, vector in embeddings.items():
                parsed_key, parsed_vector = parse_embedding_record(key, vector)
                self.embeddings[parsed_key] = parsed_vector

    def get_embeds(
        self,
        texts: Sequence[str],
        *,
        prefix: str | None = None,
    ) -> list[Embedding]:
        if prefix is not None:
            raise ValueError("CachedEmbedder does not support prefix inference")

        keys = [self._normalize_key(text) for text in texts]
        missing = [key for key in dict.fromkeys(keys) if key not in self.embeddings]
        if missing:
            raise KeyError(f"Missing cached embeddings for: {', '.join(missing)}")
        return [list(self.embeddings[key]) for key in keys]

    def get_embed_dim(self) -> int:
        return len(next(iter(self.embeddings.values())))


class OnTheFlyEmbedder(Embedder):
    """Run an embedding model at request time."""

    def __init__(
        self,
        *,
        model: Encoder | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 64,
        normalize_embeddings: bool = False,
        text_source: str = "title",
        raws: Mapping[str, str] | None = None,
        cache_inferred: bool = True,
    ) -> None:
        if text_source not in TEXT_SOURCES:
            available = ", ".join(sorted(TEXT_SOURCES))
            raise ValueError(f"text_source must be one of: {available}")
        self.model = model
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.text_source = text_source
        self.raws = dict(raws or {})
        self.cache_inferred = cache_inferred
        self._embed_cache: dict[str, Embedding] = {}

    def get_embeds(
        self,
        texts: Sequence[str],
        *,
        prefix: str | None = None,
    ) -> list[Embedding]:
        keys = [self._normalize_key(text) for text in texts]
        keys = [self._select_text(key) for key in keys]
        if prefix is not None:
            keys = [prefix + key for key in keys]
        if not self.cache_inferred:
            return self._infer(keys)

        missing = [key for key in dict.fromkeys(keys) if key not in self._embed_cache]
        if missing:
            vectors = self._infer(missing)
            self._embed_cache.update(zip(missing, vectors))
        return [list(self._embed_cache[key]) for key in keys]

    def get_embed_dim(self) -> int:
        dimension = getattr(self._get_model(), "get_sentence_embedding_dimension", None)
        if callable(dimension):
            result = dimension()
            if result is not None:
                return int(result)
        return len(self.get_embed(""))

    def _select_text(self, key: str) -> str:
        if self.text_source == "title":
            return key
        if self.text_source == "raw":
            if key not in self.raws:
                raise KeyError(f"Missing raw text for title {key!r}")
            return self.raws[key]
        return self.raws.get(key, key)

    def _infer(self, texts: list[str]) -> list[Embedding]:
        model = self._get_model()
        encoded = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        vectors = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        return [
            as_float_list(vector, name=f"inferred_embedding[{text!r}]")
            for text, vector in zip(texts, vectors)
        ]

    def _get_model(self) -> Encoder:
        if self.model is not None:
            return self.model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Embedding inference requires sentence-transformers. "
                "Install it with: pip install sentence-transformers torch"
            ) from exc

        self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    def save_cache(self, output_path: str | Path) -> int:
        """Persist inferred embeddings in the JSONL format accepted by CachedEmbedder."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        with output.open("w", encoding="utf-8") as handle:
            for key, vector in self._embed_cache.items():
                handle.write(
                    json.dumps(
                        {"title": key, "embedding": vector},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
        return written


class EmbeddingModel(Embedder):
    """Argument-driven wrapper around one embedding backend."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        config: Mapping[str, object] | None = None,
        use_cache: bool | None = None,
        embeddings_path: str | Path | None = None,
        embeddings: Mapping[str, Sequence[float]] | None = None,
        model: Encoder | None = None,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
        text_source: str | None = None,
        raws: Mapping[str, str] | None = None,
        cache_inferred: bool | None = None,
    ) -> None:
        merged_config: dict[str, object] = {}
        if config_path is not None:
            merged_config.update(load_embedding_config(config_path))
        if config is not None:
            merged_config.update(config)

        if use_cache is None:
            use_cache = bool(require_config_value(merged_config, "use_cache", merged_config.get("use_cache")))
        if embeddings_path is None:
            embeddings_path = merged_config.get("embeddings_path")

        if use_cache:
            self.backend: Embedder = CachedEmbedder(
                embeddings_path=embeddings_path if isinstance(embeddings_path, (str, Path)) else None,
                embeddings=embeddings,
            )
            return

        if model_name is None:
            model_name = str(
                require_config_value(
                    merged_config,
                    "model_name",
                    merged_config.get("model_name"),
                )
            )
        if device is None:
            device = merged_config.get("device")
        if batch_size is None:
            batch_size = int(
                require_config_value(
                    merged_config,
                    "batch_size",
                    merged_config.get("batch_size"),
                )
            )
        if normalize_embeddings is None:
            normalize_embeddings = bool(
                require_config_value(
                    merged_config,
                    "normalize_embeddings",
                    merged_config.get("normalize_embeddings"),
                )
            )
        if text_source is None:
            text_source = str(
                require_config_value(
                    merged_config,
                    "text_source",
                    merged_config.get("text_source"),
                )
            )
        if cache_inferred is None:
            cache_inferred = bool(
                require_config_value(
                    merged_config,
                    "cache_inferred",
                    merged_config.get("cache_inferred"),
                )
            )

        if embeddings_path is not None or embeddings is not None:
            raise ValueError("Set use_cache=True to use cached embeddings")

        self.backend = OnTheFlyEmbedder(
            model=model,
            model_name=model_name,
            device=str(device) if device is not None else None,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            text_source=text_source,
            raws=raws,
            cache_inferred=cache_inferred,
        )

    def get_embeds(
        self,
        texts: Sequence[str],
        *,
        prefix: str | None = None,
    ) -> list[Embedding]:
        return self.backend.get_embeds(texts, prefix=prefix)

    def get_embed_dim(self) -> int:
        return self.backend.get_embed_dim()

    def save_cache(
        self,
        keys: Sequence[str],
        *,
        output_path: str | Path,
        prefix: str | None = None,
        batch_size: int = 64,
    ) -> int:
        """Write title-keyed embeddings in the JSONL format accepted by CachedEmbedder."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        normalized_keys = [key.strip() for key in keys if key.strip()]
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        with output.open("w", encoding="utf-8") as handle:
            for start in range(0, len(normalized_keys), batch_size):
                batch_keys = normalized_keys[start : start + batch_size]
                vectors = self.get_embeds(batch_keys, prefix=prefix)
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
