from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from datasets import load_dataset

from embed import (
    DEFAULT_ON_THE_FLY_CONFIG,
    EmbeddingModel,
    load_embedding_config,
    require_config_value,
)
from environment import NamuwikiEnvironment
from utils import Title


PREFIX_ALIASES = {
    "none": None,
    "query": "query: ",
    "passage": "passage: ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate title-keyed embedding caches for Traveling Namuwiki."
    )
    parser.add_argument(
        "--embedding-config",
        default=str(DEFAULT_ON_THE_FLY_CONFIG),
        help="YAML embedding generation config path.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_raw_texts(raw_path: str, split: str, allowed_titles: set[Title]) -> dict[Title, str]:
    raws: dict[Title, str] = {}
    for row in load_dataset(raw_path, split=split):
        title = str(row.get("title") or "").strip()
        if title in allowed_titles and title not in raws:
            raws[title] = str(row.get("text") or "")
    return raws


def main() -> None:
    args = parse_args()
    config = load_embedding_config(args.embedding_config)
    use_cache = bool(require_config_value(config, "use_cache", config.get("use_cache")))
    if use_cache:
        raise ValueError("generate_embeddings requires an on-the-fly embedding config")

    actions_path = str(
        require_config_value(config, "actions_path", config.get("actions_path"))
    )
    split = str(require_config_value(config, "split", config.get("split")))
    output_path = Path(
        str(require_config_value(config, "output_path", config.get("output_path")))
    )
    text_source = str(
        require_config_value(config, "text_source", config.get("text_source"))
    )
    prefix_arg = config.get("prefix")
    prefix = None if prefix_arg is None else PREFIX_ALIASES.get(str(prefix_arg), str(prefix_arg))
    batch_size = int(
        require_config_value(config, "batch_size", config.get("batch_size"))
    )
    model_name = str(
        require_config_value(config, "model_name", config.get("model_name"))
    )
    normalize_embeddings = bool(
        require_config_value(
            config,
            "normalize_embeddings",
            config.get("normalize_embeddings"),
        )
    )

    log(
        "[generate_embeddings] "
        f"model_name={model_name} "
        f"text_source={text_source} "
        f"prefix={prefix!r}"
    )

    log("[generate_embeddings] loading environment")
    env = NamuwikiEnvironment.from_dataset(actions_path, split=split)
    titles = env.all_titles()
    log(f"[generate_embeddings] collected {len(titles)} unique titles")

    raws = {}
    if text_source != "title":
        raw_path = str(require_config_value(config, "raw_path", config.get("raw_path")))
        raws = load_raw_texts(raw_path, split, set(titles))
        log(f"[generate_embeddings] loaded raw text for {len(raws)} titles")

    embedder = EmbeddingModel(
        config_path=args.embedding_config,
        raws=raws,
    )

    log(f"[generate_embeddings] writing cache to {output_path}")
    written_titles = embedder.save_cache(
        titles,
        output_path=output_path,
        prefix=prefix,
        batch_size=batch_size,
    )

    log("[generate_embeddings] completed successfully")
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "titles": written_titles,
                "model_name": model_name,
                "text_source": text_source,
                "prefix": prefix,
                "normalize_embeddings": normalize_embeddings,
                "device": config.get("device"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
