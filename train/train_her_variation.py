"""
train_her_variation.py — HER with pluggable goal-selection strategies.

기존 HER (train_hindsight_target_a2c_v2.py) 대비 변형:

  --her-strategy future   [기존] future states에서 uniform random 샘플링
  --her-strategy final    [표준] episode 마지막 도달 state를 goal로 사용
  --her-strategy all      [분석용] 가능한 future states를 모두 사용
  --her-strategy semantic [신규] cached_similarity로 원래 target과 비슷한 goal을 우선 선택
                                  → "더 유용한" hindsight goals → 학습 효율 ↑
  --her-strategy mixed    [신규] future(semantic-weighted) + final(episode 마지막 도달점) 혼합
                                  → goal diversity 확보 + 유용한 goal 선호

Semantic-weighted HER 직관:
  에피소드 말미에 "전라북도청" 페이지에 도달했다면,
  - 원래 목표가 "서울특별시청"이면 → 전라북도청을 hindsight goal로 쓰면 별로 유용하지 않음
  - 원래 목표가 "전주시청"이면 → 전라북도청이 목표와 semantically 가까움 → 좋은 hindsight goal
  cached_similarity로 이를 판단해 유용한 goal을 더 자주 샘플링.

Mixed HER 직관:
  - final: 무조건 마지막 도달점 사용 (항상 성공하는 안전한 goal)
  - semantic-weighted future: 나머지 k-1개는 유사도 가중치로 선택
  → diversity(final) + quality(semantic) 동시 확보
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from environment import NamuwikiEnvironment
from evaluate_paths import iter_path_examples, limited
from models.networks.neural_target_actor_critic import NeuralTargetActorCritic, save_checkpoint
from similarity import cached_similarity
from utils import ACTIONS_DATASET, PATHS_DATASET

import train_hindsight_target_a2c_v2 as base

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
        description="HER with pluggable goal-selection strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--actions-path", default=ACTIONS_DATASET)
    parser.add_argument("--paths-path", default=PATHS_DATASET)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.92)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--her-k", type=int, default=2,
                        help="에피소드당 hindsight goal 샘플 수")
    parser.add_argument("--her-coef", type=float, default=0.2,
                        help="HER loss 가중치")
    parser.add_argument(
        "--her-strategy",
        choices=["future", "final", "all", "semantic", "mixed"],
        default="future",
        help=(
            "future  : uniform random (기존 방식)\n"
            "final   : episode의 마지막 도달 state를 goal로 사용\n"
            "all     : 각 transition의 모든 future state를 goal로 사용\n"
            "semantic: cached_similarity 가중치로 원래 target과 유사한 goal 우선\n"
            "mixed   : final(마지막 도달점) + semantic-weighted future 혼합"
        ),
    )
    parser.add_argument("--her-temp", type=float, default=1.0,
                        help="semantic 전략의 softmax temperature (낮을수록 고유사도 goal에 집중)")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bucket-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-output", type=Path,
                        default=Path("checkpoints") / "her_variation.pt")
    # wandb
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HER goal selection strategies
# ---------------------------------------------------------------------------

def _semantic_weights(
    candidates: list[str],
    original_target: str,
    temperature: float,
) -> list[float]:
    """
    각 후보 goal에 대해 original_target과의 lexical similarity를 계산하고
    softmax로 정규화한 가중치를 반환한다.
    temperature가 낮을수록 유사도 높은 goal에 더 집중.
    """
    sims = [cached_similarity(c, original_target) for c in candidates]
    # softmax with temperature
    max_s = max(sims) if sims else 0.0
    exps = [math.exp((s - max_s) / max(temperature, 1e-6)) for s in sims]
    total = sum(exps) + 1e-12
    return [e / total for e in exps]


def _weighted_sample_without_replacement(
    indexes: list[int],
    weights: list[float],
    k: int,
) -> list[int]:
    """Sample up to k unique indexes while respecting relative weights."""
    if k <= 0 or not indexes:
        return []

    remaining_indexes = list(indexes)
    remaining_weights = list(weights)
    sampled: list[int] = []
    for _ in range(min(k, len(remaining_indexes))):
        total = sum(remaining_weights)
        if total <= 0.0:
            choice_position = random.randrange(len(remaining_indexes))
        else:
            choice_position = random.choices(
                range(len(remaining_indexes)),
                weights=remaining_weights,
                k=1,
            )[0]
        sampled.append(remaining_indexes.pop(choice_position))
        remaining_weights.pop(choice_position)
    return sampled


def apply_hindsight_updates_varied(
    *,
    network: NeuralTargetActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[base.EpisodeTransition],
    achieved_titles: Sequence[str],
    original_target: str,
    gamma: float,
    her_k: int,
    her_coef: float,
    her_strategy: str,
    her_temp: float,
    device: torch.device,
) -> tuple[list[float], int]:
    """
    HER goal-relabeling 업데이트.
    her_strategy에 따라 goal 샘플링 방식이 달라진다.

    future  : 기존과 동일 (uniform from future states)
    final   : 마지막 도달 state를 모든 transition의 hindsight goal로 사용
    all     : transition별 future states를 전부 사용
    semantic: 원래 target과의 similarity로 가중치를 두어 샘플링
    mixed   : final(마지막 도달점) 1개 + semantic future (k-1)개
    """
    losses: list[float] = []
    update_count = 0
    if her_k <= 0 or not transitions:
        return losses, update_count

    achieved_list = list(achieved_titles)
    final_index = len(achieved_list) - 1

    for t_idx, transition in enumerate(transitions):
        future_indexes = list(range(t_idx, len(achieved_list)))
        if not future_indexes:
            continue

        # --- goal index 샘플링 ---
        if her_strategy == "future":
            # 기존 방식: uniform random
            sampled = random.sample(future_indexes, k=min(her_k, len(future_indexes)))

        elif her_strategy == "final":
            sampled = [final_index]

        elif her_strategy == "all":
            sampled = future_indexes

        elif her_strategy == "semantic":
            # 원래 target과 유사한 goal 우선 샘플링
            candidates = [achieved_list[i] for i in future_indexes]
            weights = _semantic_weights(candidates, original_target, her_temp)
            k = min(her_k, len(future_indexes))
            sampled = _weighted_sample_without_replacement(future_indexes, weights, k)

        else:  # "mixed"
            # final goal 1개 확정 + semantic-weighted future (k-1)개
            sampled_set = {final_index}
            remaining_k = her_k - 1
            if remaining_k > 0 and len(future_indexes) > 0:
                candidates = [achieved_list[i] for i in future_indexes]
                weights = _semantic_weights(candidates, original_target, her_temp)
                extra = _weighted_sample_without_replacement(
                    future_indexes,
                    weights,
                    min(remaining_k, len(future_indexes)),
                )
                sampled_set.update(extra)
            sampled = list(sampled_set)

        # --- 각 sampled goal로 HER 업데이트 ---
        for future_idx in sampled:
            hindsight_goal = achieved_list[future_idx]
            distance_to_goal = future_idx - t_idx + 1

            logits, value = base.forward_state(
                network=network,
                current=transition.current,
                target=hindsight_goal,
                actions=transition.actions,
                device=device,
            )
            target_action = torch.tensor(
                [transition.action_index], dtype=torch.long, device=device
            )
            actor_loss = F.cross_entropy(logits.unsqueeze(0), target_action)
            target_value = torch.tensor(
                gamma ** (distance_to_goal - 1), dtype=torch.float32, device=device
            )
            critic_loss = F.smooth_l1_loss(value.squeeze(), target_value)
            loss = her_coef * (actor_loss + 0.5 * critic_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            update_count += 1

    return losses, update_count


# ---------------------------------------------------------------------------
# 학습 루프 (base의 train_epoch을 HER strategy 변형으로 교체)
# ---------------------------------------------------------------------------

def train_epoch(
    *,
    env: NamuwikiEnvironment,
    network: NeuralTargetActorCritic,
    optimizer: torch.optim.Optimizer,
    examples: Sequence,
    max_steps: int,
    gamma: float,
    entropy_coef: float,
    value_coef: float,
    her_k: int,
    her_coef: float,
    her_strategy: str,
    her_temp: float,
    device: torch.device,
) -> dict:
    network.train()
    losses: list[float] = []
    her_losses: list[float] = []
    her_update_count = 0
    returns: list[float] = []
    successes = 0

    for example in examples:
        current = example.start_title
        seen: set[str] = {current}
        episode_return = 0.0
        transitions: list[base.EpisodeTransition] = []
        achieved_titles: list[str] = []

        for _ in range(max_steps):
            actions = list(env.actions(current))
            if not actions:
                episode_return += -0.2
                break

            logits, value = base.forward_state(
                network=network,
                current=current,
                target=example.target_title,
                actions=actions,
                device=device,
            )
            penalties = torch.tensor(
                [-0.5 if a in seen else 0.0 for a in actions],
                dtype=torch.float32, device=device,
            )
            dist = torch.distributions.Categorical(logits=logits + penalties)
            action_index = dist.sample()
            chosen = int(action_index.item())
            action = actions[chosen]

            next_seen = action in seen
            reached_target = action == example.target_title
            dead_end = not list(env.actions(action))

            reward = -0.05
            if reached_target:
                reward = 1.0
            elif next_seen:
                reward = -0.35
            elif dead_end:
                reward = -0.2

            done = reached_target or next_seen or dead_end

            with torch.no_grad():
                next_value = torch.zeros((), dtype=torch.float32, device=device)
                if not done:
                    nxt_actions = list(env.actions(action)) or [action]
                    _, nv = base.forward_state(
                        network=network,
                        current=action,
                        target=example.target_title,
                        actions=nxt_actions,
                        device=device,
                    )
                    next_value = nv.squeeze()
                td_target = torch.tensor(reward, dtype=torch.float32,
                                         device=device) + gamma * next_value

            advantage = td_target - value.squeeze()
            policy_loss = -dist.log_prob(action_index) * advantage.detach()
            value_loss = F.smooth_l1_loss(value.squeeze(), td_target)
            entropy_loss = -dist.entropy()
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            episode_return += reward
            transitions.append(base.EpisodeTransition(current, actions, chosen, action))
            achieved_titles.append(action)
            if reached_target:
                successes += 1
            if done:
                break

            seen.add(action)
            current = action

        if transitions:
            episode_her_losses, episode_her_updates = apply_hindsight_updates_varied(
                network=network,
                optimizer=optimizer,
                transitions=transitions,
                achieved_titles=achieved_titles,
                original_target=example.target_title,
                gamma=gamma,
                her_k=her_k,
                her_coef=her_coef,
                her_strategy=her_strategy,
                her_temp=her_temp,
                device=device,
            )
            her_losses.extend(episode_her_losses)
            her_update_count += episode_her_updates
        returns.append(episode_return)

    return {
        "train_loss": sum(losses) / len(losses) if losses else 0.0,
        "hindsight_loss": sum(her_losses) / len(her_losses) if her_losses else 0.0,
        "hindsight_updates": her_update_count,
        "avg_return": sum(returns) / len(returns),
        "train_success_rate": successes / len(examples),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_wandb = _WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        _wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"her_{args.her_strategy}",
            config={k: v for k, v in vars(args).items()
                    if not k.startswith("wandb")},
        )
    elif args.wandb_project and not _WANDB_AVAILABLE:
        print("WARNING: wandb not installed — logging disabled.", flush=True)

    print(json.dumps({
        "device": str(device), "algo": "her_variation",
        "her_strategy": args.her_strategy, "her_k": args.her_k,
        "her_temp": args.her_temp, "wandb": use_wandb,
    }), flush=True)

    env = NamuwikiEnvironment.from_dataset(args.actions_path)
    examples = list(limited(
        iter_path_examples(args.paths_path, args.train_split), args.train_limit
    ))
    eval_examples = list(limited(
        iter_path_examples(args.paths_path, args.eval_split), args.eval_limit
    ))
    if not examples:
        raise ValueError("No training examples found.")

    network = NeuralTargetActorCritic(
        bucket_size=args.bucket_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)

    best_success = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        train_metrics = train_epoch(
            env=env,
            network=network,
            optimizer=optimizer,
            examples=examples,
            max_steps=args.max_steps,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            her_k=args.her_k,
            her_coef=args.her_coef,
            her_strategy=args.her_strategy,
            her_temp=args.her_temp,
            device=device,
        )
        eval_metrics = base.evaluate_policy(
            env=env,
            network=network,
            examples=eval_examples,
            max_steps=args.max_steps,
            device=device,
        )
        record = {
            "epoch": epoch,
            "her_strategy": args.her_strategy,
            **train_metrics,
            **eval_metrics,
        }
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
               "best_success_rate": best_success,
               "her_strategy": args.her_strategy,
               "history": history}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if use_wandb:
        _wandb.finish()


if __name__ == "__main__":
    main()
