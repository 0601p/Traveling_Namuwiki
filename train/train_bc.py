"""
train_bc.py — Standalone Behavioral Cloning from gold paths.

데이터셋의 'paths' 필드(실제 최단경로 시퀀스)를 사용해
(current, target, actions) → correct_action 을 cross-entropy로 학습.

v6의 --algo bc는 BC pretrain 후 PPO를 돌리지만,
이 스크립트는 BC만 독립적으로 학습/평가한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from typing import Sequence

import torch
import torch.nn.functional as F

from environment import NamuwikiEnvironment
from evaluate_paths import iter_path_examples, limited
from models.networks.neural_target_actor_critic import NeuralTargetActorCritic, save_checkpoint
from utils import ACTIONS_DATASET, PATHS_DATASET

# v6에서 핵심 함수들 재사용
import train_greedy_residual_ppo_v6 as v6

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning from gold paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--actions-path", default=ACTIONS_DATASET)
    parser.add_argument("--paths-path", default=PATHS_DATASET)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--bc-samples-mult", type=int, default=4,
                        help="train_limit × bc_samples_mult 개의 BC 샘플을 수집")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--prior-alpha", type=float, default=0.0,
                        help="lexical residual prior 강도 (0=pure BC)")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bucket-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-output", type=Path,
                        default=Path("checkpoints") / "bc.pt")
    # wandb
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------

def train_epoch(
    *,
    network: NeuralTargetActorCritic,
    optimizer: torch.optim.Optimizer,
    samples: Sequence[v6.BCSample],
    prior_alpha: float,
    device: torch.device,
) -> dict:
    """한 epoch의 BC 학습. cross-entropy loss로 gold action을 모방."""
    network.train()
    shuffled = list(samples)
    random.shuffle(shuffled)

    losses: list[float] = []
    correct = 0

    for sample in shuffled:
        logits, _ = v6.forward_state(
            network=network,
            current=sample.current,
            target=sample.target,
            actions=sample.actions,
            prior_alpha=prior_alpha,
            device=device,
        )
        label = torch.tensor([sample.action_index], dtype=torch.long, device=device)
        loss = F.cross_entropy(logits.unsqueeze(0), label)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        correct += int(torch.argmax(logits).item() == sample.action_index)

    n = len(samples)
    return {
        "bc_loss": sum(losses) / len(losses) if losses else 0.0,
        "bc_acc": correct / n if n else 0.0,
        "bc_samples": n,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb 초기화
    use_wandb = _WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        _wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or "bc",
            config={k: v for k, v in vars(args).items()
                    if not k.startswith("wandb")},
        )
    elif args.wandb_project and not _WANDB_AVAILABLE:
        print("WARNING: wandb not installed — logging disabled.", flush=True)

    print(json.dumps({"device": str(device), "algo": "bc",
                      "prior_alpha": args.prior_alpha, "wandb": use_wandb}),
          flush=True)

    # 환경 & 데이터
    env = NamuwikiEnvironment.from_dataset(args.actions_path)
    eval_examples = list(limited(
        iter_path_examples(args.paths_path, args.eval_split), args.eval_limit
    ))

    # BC 샘플 수집 (gold paths 사용)
    bc_samples = list(limited(
        v6.iter_bc_samples(env=env, dataset_path=args.paths_path, split=args.train_split),
        args.train_limit * args.bc_samples_mult,
    ))
    print(json.dumps({"bc_samples_total": len(bc_samples)}), flush=True)
    if not bc_samples:
        raise ValueError("No BC samples found — dataset에 'paths' 필드가 없을 수 있음.")

    # 모델 & 옵티마이저
    network = NeuralTargetActorCritic(
        bucket_size=args.bucket_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)

    best_success = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            network=network,
            optimizer=optimizer,
            samples=bc_samples,
            prior_alpha=args.prior_alpha,
            device=device,
        )
        eval_metrics = v6.evaluate_policy(
            env=env,
            network=network,
            examples=eval_examples,
            max_steps=args.max_steps,
            prior_alpha=args.prior_alpha,
            device=device,
        )
        record = {"epoch": epoch, **train_metrics, **eval_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

        if use_wandb:
            _wandb.log({k: v for k, v in record.items()
                        if isinstance(v, (int, float))}, step=epoch)

        if eval_metrics["success_rate"] > best_success:
            best_success = eval_metrics["success_rate"]
            save_checkpoint(path=args.checkpoint_output, network=network,
                            metrics=record)
            if use_wandb:
                _wandb.summary["best_success_rate"] = best_success
                _wandb.summary["best_epoch"] = epoch

    summary = {"checkpoint": str(args.checkpoint_output),
               "best_success_rate": best_success, "history": history}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if use_wandb:
        _wandb.finish()


if __name__ == "__main__":
    main()
