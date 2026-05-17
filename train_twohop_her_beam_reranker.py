from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from environment import NamuwikiEnvironment
from evaluate_beam_search import beam_search
from evaluate_paths import PathExample, iter_path_examples, limited
from models.beam_her_reranker import BeamHerReranker, save_beam_reranker_checkpoint
from utils import ACTIONS_DATASET, PATHS_DATASET, cached_similarity


@dataclass(frozen=True)
class RankingExample:
    current: str
    target: str
    actions: list[str]
    positive_index: int
    visited: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a HER-style reranker for two-hop beam candidates."
    )
    parser.add_argument("--actions-path", default=ACTIONS_DATASET)
    parser.add_argument("--paths-path", default=PATHS_DATASET)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--expand-top-k", type=int, default=16)
    parser.add_argument("--two-hop-weight", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--her-goals-per-state", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bucket-size", type=int, default=50000)
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("checkpoints") / "twohop_her_beam_reranker.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"device": str(device), "approach": "twohop_her_beam_reranker"}), flush=True)

    env = NamuwikiEnvironment.from_dataset(args.actions_path)
    train_examples = list(
        limited(iter_path_examples(args.paths_path, args.train_split), args.train_limit)
    )
    eval_examples = list(
        limited(iter_path_examples(args.paths_path, args.eval_split), args.eval_limit)
    )
    ranking_examples = build_ranking_examples(
        env=env,
        examples=train_examples,
        max_steps=args.max_steps,
        beam_size=args.beam_size,
        expand_top_k=args.expand_top_k,
        two_hop_weight=args.two_hop_weight,
        her_goals_per_state=args.her_goals_per_state,
    )
    print(
        json.dumps(
            {
                "train_paths": len(train_examples),
                "ranking_examples": len(ranking_examples),
            }
        ),
        flush=True,
    )

    model = BeamHerReranker(
        bucket_size=args.bucket_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_success = -1.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(ranking_examples)
        train_metrics = train_epoch(
            env=env,
            model=model,
            optimizer=optimizer,
            ranking_examples=ranking_examples,
            device=device,
        )
        eval_metrics = evaluate_reranked_beam(
            env=env,
            model=model,
            examples=eval_examples,
            max_steps=args.max_steps,
            beam_size=args.beam_size,
            expand_top_k=args.expand_top_k,
            two_hop_weight=args.two_hop_weight,
            device=device,
        )
        record = {"epoch": epoch, **train_metrics, **eval_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if eval_metrics["success_rate"] > best_success:
            best_success = eval_metrics["success_rate"]
            save_beam_reranker_checkpoint(
                path=args.checkpoint_output,
                model=model,
                metrics=record,
            )

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint_output),
                "best_success_rate": best_success,
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def build_ranking_examples(
    *,
    env: NamuwikiEnvironment,
    examples: Sequence[PathExample],
    max_steps: int,
    beam_size: int,
    expand_top_k: int,
    two_hop_weight: float,
    her_goals_per_state: int,
) -> list[RankingExample]:
    ranking_examples: list[RankingExample] = []
    for example in examples:
        result = beam_search(
            env=env,
            start=example.start_title,
            target=example.target_title,
            method="twohop",
            beam_size=beam_size,
            expand_top_k=expand_top_k,
            max_steps=max_steps,
            lexical_weight=1.0,
            two_hop_weight=two_hop_weight,
            lm_weight=0.5,
            value_weight=0.0,
            policy_weight=0.0,
            cycle_penalty=0.5,
            lm_scorer=None,
            value_scorer=None,
            reranker_scorer=None,
            reranker_weight=0.0,
        )
        visited = result.visited
        for index, current in enumerate(visited[:-1]):
            next_title = visited[index + 1]
            actions = ranked_actions(
                env=env,
                current=current,
                target=example.target_title,
                expand_top_k=expand_top_k,
            )
            if next_title not in actions:
                continue
            positive_index = actions.index(next_title)
            goals = [example.target_title] if result.reached_target else []
            achieved_suffix = visited[index + 1 :]
            if achieved_suffix:
                goals.extend(random.sample(achieved_suffix, k=min(her_goals_per_state, len(achieved_suffix))))
            for goal in dict.fromkeys(goals):
                ranking_examples.append(
                    RankingExample(
                        current=current,
                        target=goal,
                        actions=actions,
                        positive_index=positive_index,
                        visited=tuple(visited[: index + 1]),
                    )
                )
    return ranking_examples


def ranked_actions(
    *,
    env: NamuwikiEnvironment,
    current: str,
    target: str,
    expand_top_k: int,
) -> list[str]:
    return sorted(
        list(env.actions(current)),
        key=lambda action: cached_similarity(action, target),
        reverse=True,
    )[:expand_top_k]


def scalar_features(
    *,
    env: NamuwikiEnvironment,
    current: str,
    target: str,
    actions: Sequence[str],
    visited: Sequence[str],
    two_hop_weight: float,
    device: torch.device,
) -> torch.Tensor:
    current_similarity = cached_similarity(current, target)
    seen = set(visited)
    rows: list[list[float]] = []
    for action in actions:
        lexical = cached_similarity(action, target)
        neighbors = list(env.actions(action))
        two_hop = 0.0
        if neighbors:
            two_hop = max(cached_similarity(neighbor, target) for neighbor in neighbors[:64])
        rows.append(
            [
                lexical,
                two_hop_weight * two_hop,
                lexical - current_similarity,
                math.log1p(len(neighbors)) / 8.0,
                1.0 if action in seen else 0.0,
                1.0 if action == target else 0.0,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32, device=device)


def train_epoch(
    *,
    env: NamuwikiEnvironment,
    model: BeamHerReranker,
    optimizer: torch.optim.Optimizer,
    ranking_examples: Sequence[RankingExample],
    device: torch.device,
) -> dict:
    model.train()
    losses: list[float] = []
    correct = 0
    for example in ranking_examples:
        features = scalar_features(
            env=env,
            current=example.current,
            target=example.target,
            actions=example.actions,
            visited=example.visited,
            two_hop_weight=0.4,
            device=device,
        )
        scores = model.scores(
            current=example.current,
            target=example.target,
            actions=example.actions,
            scalar_features=features,
            device=device,
        )
        target = torch.tensor([example.positive_index], dtype=torch.long, device=device)
        loss = F.cross_entropy(scores.unsqueeze(0), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if int(torch.argmax(scores).item()) == example.positive_index:
            correct += 1
    return {
        "train_loss": sum(losses) / len(losses) if losses else 0.0,
        "train_acc": correct / len(ranking_examples) if ranking_examples else 0.0,
    }


@torch.no_grad()
def evaluate_reranked_beam(
    *,
    env: NamuwikiEnvironment,
    model: BeamHerReranker,
    examples: Sequence[PathExample],
    max_steps: int,
    beam_size: int,
    expand_top_k: int,
    two_hop_weight: float,
    device: torch.device,
) -> dict:
    from evaluate_beam_search import beam_search

    model.eval()
    successes = 0
    steps: list[int] = []
    for example in examples:
        result = beam_search(
            env=env,
            start=example.start_title,
            target=example.target_title,
            method="twohop_her_reranker",
            beam_size=beam_size,
            expand_top_k=expand_top_k,
            max_steps=max_steps,
            lexical_weight=1.0,
            two_hop_weight=two_hop_weight,
            lm_weight=0.5,
            value_weight=0.0,
            policy_weight=0.0,
            cycle_penalty=0.5,
            lm_scorer=None,
            value_scorer=None,
            reranker_scorer=model,
            reranker_weight=0.25,
        )
        if result.reached_target:
            successes += 1
            steps.append(result.steps_taken)
    return {
        "success_rate": successes / len(examples) if examples else 0.0,
        "mean_steps_on_success": sum(steps) / len(steps) if steps else None,
        "count": len(examples),
    }


if __name__ == "__main__":
    main()
