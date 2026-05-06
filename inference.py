from __future__ import annotations

import argparse
import json

from environment import NamuwikiEnvironment
from models import create_model
from utils import ACTIONS_DATASET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore Namuwiki with an environment and a model."
    )
    parser.add_argument(
        "--actions-path",
        default=ACTIONS_DATASET,
        help="Dataset id or local dataset path passed to load_dataset().",
    )
    parser.add_argument("--start-title", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", default="randomwalk")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--allow-cycle",
        dest="stop_on_cycle",
        action="store_false",
        help="Keep walking even if a title is revisited.",
    )
    parser.set_defaults(stop_on_cycle=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = NamuwikiEnvironment.from_dataset(args.actions_path)
    model = create_model(args.model)
    result = env.walk(
        start_title=args.start_title,
        target_title=args.target,
        model=model,
        max_steps=args.max_steps,
        stop_on_cycle=args.stop_on_cycle,
    )

    print(
        json.dumps(
            {
                "start_title": result.start_title,
                "target_title": result.target_title,
                "end_title": result.end_title,
                "steps_taken": result.steps_taken,
                "visited": result.visited,
                "stopped_reason": result.stopped_reason,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
