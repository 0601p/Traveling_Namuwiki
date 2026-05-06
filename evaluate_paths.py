from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from environment import NamuwikiEnvironment
from models import available_models, create_model
from utils import ACTIONS_DATASET, PATHS_DATASET, Title


@dataclass(frozen=True)
class PathExample:
    start_title: Title
    target_title: Title
    hop: int

    @property
    def min_distance(self) -> int:
        return self.hop


def row_to_example(row: dict) -> PathExample:
    start_title = str(row.get("start_title") or "").strip()
    target_title = str(row.get("target_title") or "").strip()
    if not start_title or not target_title:
        raise ValueError(f"Missing start_title or target_title: {row}")

    return PathExample(start_title, target_title, hop=parse_hop(row["hop"]))


def parse_hop(value: object) -> int:
    hop = int(value)
    if hop <= 0:
        raise ValueError(f"hop must be positive: {value}")
    return hop


def iter_path_examples(dataset_path: str, split: str) -> Iterable[PathExample]:
    dataset = load_dataset(dataset_path, split=split)
    for row in dataset:
        yield row_to_example(row)


def limited(examples: Iterable[PathExample], limit: int | None) -> Iterable[PathExample]:
    for index, example in enumerate(examples):
        if limit is not None and index >= limit:
            break
        yield example


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def evaluate(args: argparse.Namespace) -> dict:
    env = NamuwikiEnvironment.from_dataset(args.actions_path)
    model = create_model(args.model)
    failure_distance = (
        args.failure_distance
        if args.failure_distance is not None
        else args.max_steps + 1
    )

    distances_by_gold: dict[int, list[float]] = defaultdict(list)
    counts_by_gold: Counter[int] = Counter()
    success_by_gold: Counter[int] = Counter()
    stopped_reasons: Counter[str] = Counter()
    total = 0

    predictions_file = None
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions_file = args.predictions_output.open("w", encoding="utf-8")

    try:
        examples = iter_path_examples(args.paths_path, args.split)
        for example in limited(examples, args.limit):
            gold_distance = example.min_distance
            if gold_distance > args.max_gold_distance:
                continue

            total += 1
            counts_by_gold[gold_distance] += 1
            result = env.walk(
                start_title=example.start_title,
                target_title=example.target_title,
                model=model,
                max_steps=args.max_steps,
                stop_on_cycle=args.stop_on_cycle,
            )

            stopped_reasons[result.stopped_reason] += 1
            if result.reached_target:
                model_distance = result.steps_taken
                success_by_gold[gold_distance] += 1
            else:
                model_distance = failure_distance

            distances_by_gold[gold_distance].append(float(model_distance))

            if predictions_file is not None:
                predictions_file.write(
                    json.dumps(
                        {
                            "start_title": example.start_title,
                            "target_title": example.target_title,
                            "gold_min_distance": gold_distance,
                            "model_distance": model_distance,
                            "reached": result.reached_target,
                            "stopped_reason": result.stopped_reason,
                            "visited": result.visited,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if predictions_file is not None:
            predictions_file.close()

    score_by_min_distance = {
        distance: mean(distances_by_gold[distance])
        for distance in range(1, args.max_gold_distance + 1)
        if counts_by_gold[distance]
    }
    success_rate_by_min_distance = {
        distance: success_by_gold[distance] / counts_by_gold[distance]
        for distance in range(1, args.max_gold_distance + 1)
        if counts_by_gold[distance]
    }

    return {
        "split": args.split,
        "model": args.model,
        "total": total,
        "predictions_output": str(args.predictions_output),
        "score_by_min_distance": score_by_min_distance,
        "success_rate_by_min_distance": success_rate_by_min_distance,
        "count_by_min_distance": dict(counts_by_gold),
        "failure_distance": failure_distance,
        "stopped_reasons": dict(stopped_reasons),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on Traveling_Namuwiki_Paths."
    )
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    parser.add_argument(
        "--model",
        default="randomwalk",
        choices=available_models(),
        help="Model used to sample actions.",
    )
    parser.add_argument(
        "--actions-path",
        default=ACTIONS_DATASET,
        help="Dataset id or local dataset path passed to load_dataset().",
    )
    parser.add_argument(
        "--paths-path",
        default=PATHS_DATASET,
        help="Dataset id or local dataset path passed to load_dataset().",
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-gold-distance", type=int, default=10)
    parser.add_argument(
        "--failure-distance",
        type=float,
        help="Distance assigned when the model fails to reach the target.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-cycle",
        dest="stop_on_cycle",
        action="store_false",
        help="Keep walking even if a title is revisited.",
    )
    parser.set_defaults(stop_on_cycle=True)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        help="Per-example JSONL output path.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Summary metrics JSON output path.",
    )
    args = parser.parse_args()
    if args.predictions_output is None:
        args.predictions_output = Path("outputs") / (
            f"{args.split}_{args.model}_predictions.jsonl"
        )
    if args.metrics_output is None:
        args.metrics_output = Path("outputs") / f"{args.split}_{args.model}_metrics.json"
    return args


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
