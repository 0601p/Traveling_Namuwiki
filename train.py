from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from utils import ACTIONS_DATASET


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
        "--algorithm",
        default="basic_reinforce",
        help="Registered training procedure, e.g. basic_reinforce.",
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

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10)
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
        default=Path("outputs") / "ar_walk.pt",
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
        default=Path("outputs") / "train_ar_walk_metrics.json",
        help="Where to write the training summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from environment import NamuwikiEnvironment
    from models.ar_walk import AutoregressiveWalk, RandomPairTaskSampler

    set_seed(args.seed)
    device = resolve_device(args.device)

    env = NamuwikiEnvironment.from_dataset(
        args.actions_path,
        split=args.actions_split,
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

    sampler = RandomPairTaskSampler(
        env.graph,
        seed=args.seed,
        allow_same_title=args.allow_same_title,
    )
    config = build_config(args)
    summary = model.train_policy(
        env.graph,
        algorithm=args.algorithm,
        config=config,
        sampler=sampler,
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
        "device": device,
        "checkpoint_output": checkpoint_output,
        "summary": asdict(summary),
        "config": asdict(config),
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
