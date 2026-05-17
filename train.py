from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from utils import ACTIONS_DATASET, PATHS_DATASET, Title


def default_run_dir() -> Path:
    """Return a unique timestamped output directory for this training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / timestamp


def resolve_device(device: str) -> str:
    """Resolve the requested training device."""
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int | None) -> None:
    """Seed local RNGs used by the trainer."""
    if seed is None:
        return
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_path_endpoint_titles(dataset_path: str, split: str) -> set[Title]:
    """Load all start/target titles from one paths split."""
    from datasets import load_dataset

    endpoint_titles: set[Title] = set()
    for row in load_dataset(dataset_path, split=split):
        start_title = str(row.get("start_title") or "").strip()
        target_title = str(row.get("target_title") or "").strip()
        if start_title:
            endpoint_titles.add(start_title)
        if target_title:
            endpoint_titles.add(target_title)
    return endpoint_titles


def filter_graph_titles(
    graph: dict[Title, list[Title]],
    excluded_titles: set[Title],
) -> dict[Title, list[Title]]:
    """Remove held-out titles from training states and outgoing actions."""
    if not excluded_titles:
        return graph

    return {
        title: [action for action in actions if action not in excluded_titles]
        for title, actions in graph.items()
        if title not in excluded_titles
    }


def build_config(args: argparse.Namespace) -> ReinforceConfig:
    """Translate CLI flags into the basic REINFORCE config."""
    from models.ar_walk import ReinforceConfig

    return ReinforceConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        baseline_decay=args.baseline_decay,
        success_reward=args.success_reward,
        failure_penalty=args.failure_penalty,
        step_penalty=args.step_penalty,
        dead_end_penalty=args.dead_end_penalty,
        cycle_penalty=args.cycle_penalty,
        max_grad_norm=args.max_grad_norm,
        temperature=args.temperature,
        stop_on_cycle=args.stop_on_cycle,
        log_every=args.log_every,
        hindsight_goals_per_episode=args.hindsight_goals_per_episode,
        hindsight_loss_weight=args.hindsight_loss_weight,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AutoregressiveWalk without gold shortest paths."
    )
    parser.add_argument(
        "--actions-path",
        default=ACTIONS_DATASET,
        help="Dataset id or local dataset path passed to load_dataset().",
    )
    parser.add_argument(
        "--actions-split",
        default="train",
        help="Split used from the actions dataset.",
    )
    parser.add_argument(
        "--paths-path",
        default=PATHS_DATASET,
        help="Paths dataset id or local path used for endpoint leakage checks.",
    )
    parser.add_argument(
        "--exclude-path-endpoints-split",
        choices=["train", "validation", "test"],
        default=None,
        help="Exclude start/target titles from this paths split during training.",
    )
    parser.add_argument(
        "--algorithm",
        default="basic_reinforce",
        help="Registered training procedure, e.g. basic_reinforce or her.",
    )

    parser.add_argument("--encoder-name", default="google/embeddinggemma-300m")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff-dim", type=int)
    parser.add_argument("--cache-size", type=int, default=8192)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, cuda:0, mps, etc.",
    )

    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--task-sampler",
        choices=["reachable", "random"],
        default="reachable",
        help="Use reachable short-walk tasks for training, or uniform random pairs.",
    )
    parser.add_argument(
        "--reachable-depth",
        type=int,
        default=None,
        help="Maximum random-walk depth for reachable task sampling. Defaults to max_steps.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--baseline-decay", type=float, default=0.95)
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--failure-penalty", type=float, default=-0.2)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--dead-end-penalty", type=float, default=-0.1)
    parser.add_argument("--cycle-penalty", type=float, default=-0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--hindsight-goals-per-episode",
        type=int,
        default=4,
        help="Number of achieved pages to relabel as HER goals per rollout.",
    )
    parser.add_argument(
        "--hindsight-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier applied to relabeled HER losses.",
    )
    parser.add_argument(
        "--allow-cycle",
        dest="stop_on_cycle",
        action="store_false",
        help="Keep an episode running after revisiting a title.",
    )
    parser.set_defaults(stop_on_cycle=True)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-same-title",
        action="store_true",
        help="Allow sampled tasks where start and target are identical.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Existing checkpoint to load before training.",
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=None,
        help="Where to save the trained policy checkpoint.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run training without writing a checkpoint.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Where to write the training summary JSON.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training metrics to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default="traveling-namuwiki")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = default_run_dir()
    if args.checkpoint_output is None:
        args.checkpoint_output = run_dir / "ar_walk.pt"
    if args.metrics_output is None:
        args.metrics_output = run_dir / "train_ar_walk_metrics.json"

    from environment import NamuwikiEnvironment
    from models.ar_walk import (
        AutoregressiveWalk,
        RandomPairTaskSampler,
        RandomWalkTaskSampler,
    )

    set_seed(args.seed)
    device = resolve_device(args.device)

    env = NamuwikiEnvironment.from_dataset(
        args.actions_path,
        split=args.actions_split,
    )
    train_graph = env.graph
    excluded_endpoint_titles: set[Title] = set()
    if args.exclude_path_endpoints_split is not None:
        excluded_endpoint_titles = load_path_endpoint_titles(
            args.paths_path,
            args.exclude_path_endpoints_split,
        )
        action_titles = set(env.graph).union(
            action for actions in env.graph.values() for action in actions
        )
        overlapping_titles = action_titles & excluded_endpoint_titles
        train_graph = filter_graph_titles(env.graph, excluded_endpoint_titles)
        print(
            {
                "excluded_path_endpoints_split": args.exclude_path_endpoints_split,
                "excluded_endpoint_titles": len(excluded_endpoint_titles),
                "overlap_with_action_graph_titles": len(overlapping_titles),
                "training_graph_pages_before": len(env.graph),
                "training_graph_pages_after": len(train_graph),
            }
        )

    model = AutoregressiveWalk(
        encoder_name=args.encoder_name,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        ff_dim=args.ff_dim,
        cache_size=args.cache_size,
        device=device,
    )
    if args.resume_checkpoint is not None:
        model.load_checkpoint(args.resume_checkpoint)

    if args.task_sampler == "random":
        sampler = RandomPairTaskSampler(
            train_graph,
            seed=args.seed,
            allow_same_title=args.allow_same_title,
        )
    else:
        sampler = RandomWalkTaskSampler(
            train_graph,
            max_depth=args.reachable_depth or args.max_steps,
            seed=args.seed,
            allow_same_title=args.allow_same_title,
        )
    config = build_config(args)
    wandb_run = None
    log_callback = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb logging was requested, but the wandb package is not installed."
            ) from exc

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            dir=str(run_dir),
            config={
                "model": "ar_walk",
                "algorithm": args.algorithm,
                "actions_path": args.actions_path,
                "actions_split": args.actions_split,
                "paths_path": args.paths_path,
                "excluded_path_endpoints_split": args.exclude_path_endpoints_split,
                "excluded_endpoint_titles": len(excluded_endpoint_titles),
                "device": device,
                "task_sampler": args.task_sampler,
                "reachable_depth": args.reachable_depth or args.max_steps,
                "training_graph_pages": len(train_graph),
                "reinforce": asdict(config),
            },
        )

        def log_to_wandb(payload: dict[str, float | int | str | None]) -> None:
            episode = int(payload["episode"])
            wandb.log(
                {
                    f"train/{key}": value
                    for key, value in payload.items()
                    if key != "episode"
                },
                step=episode,
            )

        log_callback = log_to_wandb

    summary = model.train_policy(
        train_graph,
        algorithm=args.algorithm,
        config=config,
        sampler=sampler,
        log_callback=log_callback,
    )

    checkpoint_output = None
    if not args.no_save:
        model.save_checkpoint(args.checkpoint_output)
        checkpoint_output = str(args.checkpoint_output)

    metrics = {
        "model": "ar_walk",
        "algorithm": args.algorithm,
        "actions_path": args.actions_path,
        "actions_split": args.actions_split,
        "paths_path": args.paths_path,
        "excluded_path_endpoints_split": args.exclude_path_endpoints_split,
        "excluded_endpoint_titles": len(excluded_endpoint_titles),
        "device": device,
        "task_sampler": args.task_sampler,
        "reachable_depth": args.reachable_depth or args.max_steps,
        "checkpoint_output": checkpoint_output,
        "summary": asdict(summary),
        "config": asdict(config),
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if wandb_run is not None:
        wandb_run.summary.update(asdict(summary))
        if checkpoint_output is not None:
            wandb_run.summary["checkpoint_output"] = checkpoint_output
        wandb_run.finish()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
