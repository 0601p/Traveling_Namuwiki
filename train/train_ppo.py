"""
train_ppo.py — PPO with Generalized Advantage Estimation (GAE).

v6의 PPO에서 개선된 점:
  1. GAE (Generalized Advantage Estimation, λ 파라미터)
     기존 v6: advantage = discounted_return - value  (simple TD)
     여기서: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
             A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
     → bias-variance tradeoff 조절 가능 (λ=0: TD, λ=1: MC)
  2. BC 없음, SIL 없음 — 순수 PPO
  3. prior_alpha 기본값 0.0 (lexical residual 미적용)
  4. wandb logging
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from environment import NamuwikiEnvironment
from evaluate_paths import iter_path_examples, limited
from models.networks.neural_target_actor_critic import NeuralTargetActorCritic, save_checkpoint
from utils import ACTIONS_DATASET, PATHS_DATASET

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
        description="PPO with GAE.",
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
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE λ. 0=pure TD, 1=pure MC return.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-eps", type=float, default=0.1,
                        help="PPO clip ε.")
    parser.add_argument("--ppo-epochs", type=int, default=2,
                        help="rollout 1회당 PPO 업데이트 반복 횟수.")
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--no-normalize-advantages", dest="normalize_advantages",
                        action="store_false",
                        help="Disable PPO advantage normalization.")
    parser.add_argument("--prior-alpha", type=float, default=0.0,
                        help="lexical residual prior 강도 (0=pure PPO, 5=residual PPO)")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bucket-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-output", type=Path,
                        default=Path("checkpoints") / "ppo.pt")
    # wandb
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GAE rollout 수집
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transition:
    current: str
    target: str
    actions: list[str]
    action_penalties: list[float]
    action_index: int
    old_log_prob: float
    value: float
    reward: float
    ret: float       # GAE return (advantage + value baseline)
    advantage: float # GAE advantage


def collect_rollouts_gae(
    *,
    env: NamuwikiEnvironment,
    network: NeuralTargetActorCritic,
    examples: Sequence,
    max_steps: int,
    gamma: float,
    gae_lambda: float,
    prior_alpha: float,
    device: torch.device,
) -> tuple[list[Transition], dict]:
    """
    에피소드를 수집하고 GAE로 advantage를 계산한다.

    GAE(λ):
      δ_t = r_t + γ · V(s_{t+1}) · (1-done_t) - V(s_t)
      A_t = δ_t + (γλ) · δ_{t+1} · (1-done_t) + ...

    s_{t+1}의 value를 사용하므로, 에피소드 수집 후 역방향으로 계산.
    terminal step: next_value = 0 (done=True이면 bootstrapping 없음)
    """
    network.eval()
    all_transitions: list[Transition] = []
    successes = 0
    returns: list[float] = []

    for example in examples:
        episode_steps: list[dict] = []
        episode_return = 0.0
        current = example.start_title
        seen = {current}

        for _ in range(max_steps):
            actions = list(env.actions(current))
            if not actions:
                episode_return += -0.2
                break

            with torch.no_grad():
                logits, value = v6.forward_state(
                    network=network,
                    current=current,
                    target=example.target_title,
                    actions=actions,
                    prior_alpha=prior_alpha,
                    device=device,
                )
            action_penalties = [-0.5 if action in seen else 0.0 for action in actions]
            penalty_tensor = torch.tensor(
                action_penalties, dtype=torch.float32, device=device
            )
            dist = torch.distributions.Categorical(logits=logits + penalty_tensor)
            action_index = dist.sample()
            action = actions[int(action_index.item())]

            reached_target = action == example.target_title
            next_seen = action in seen
            dead_end = not list(env.actions(action))
            done = reached_target or next_seen or dead_end

            reward = -0.05
            if reached_target:
                reward = 1.0
                successes += 1
            elif next_seen:
                reward = -0.35
            elif dead_end:
                reward = -0.2

            episode_steps.append({
                "current": current,
                "target": example.target_title,
                "actions": actions,
                "action_penalties": action_penalties,
                "action_index": int(action_index.item()),
                "old_log_prob": float(dist.log_prob(action_index).detach().cpu()),
                "value": float(value.squeeze().detach().cpu()),
                "reward": reward,
                "done": done,
            })
            episode_return += reward

            if done:
                break
            seen.add(action)
            current = action

        # --- GAE 역방향 계산 ---
        # terminal 이후 next_value = 0, gae = 0
        next_value = 0.0
        gae = 0.0
        for step in reversed(episode_steps):
            # done이면 bootstrap 없음(next_value=0), 아니면 다음 스텝의 value 사용
            nv = 0.0 if step["done"] else next_value
            delta = step["reward"] + gamma * nv - step["value"]
            gae = delta + gamma * gae_lambda * (0.0 if step["done"] else gae)
            step["advantage"] = gae
            step["ret"] = gae + step["value"]
            next_value = step["value"]

        for step in episode_steps:
            all_transitions.append(Transition(
                current=step["current"],
                target=step["target"],
                actions=step["actions"],
                action_penalties=step["action_penalties"],
                action_index=step["action_index"],
                old_log_prob=step["old_log_prob"],
                value=step["value"],
                reward=step["reward"],
                ret=step["ret"],
                advantage=step["advantage"],
            ))
        returns.append(episode_return)

    return all_transitions, {
        "rollout_return": sum(returns) / len(returns) if returns else 0.0,
        "train_success_rate": successes / len(examples) if examples else 0.0,
        "transitions": len(all_transitions),
    }


def update_ppo(
    *,
    network: NeuralTargetActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    prior_alpha: float,
    clip_eps: float,
    ppo_epochs: int,
    entropy_coef: float,
    value_coef: float,
    normalize_advantages: bool,
    device: torch.device,
) -> dict:
    """Run PPO updates with the same revisit penalties used during rollout."""
    network.train()
    if not transitions:
        return {"ppo_loss": 0.0, "advantage_mean": 0.0, "advantage_std": 0.0}

    raw_advantages = torch.tensor(
        [transition.advantage for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    advantage_mean = float(raw_advantages.mean().detach().cpu())
    advantage_std = float(raw_advantages.std(unbiased=False).detach().cpu())
    if normalize_advantages and raw_advantages.numel() > 1 and advantage_std > 1e-8:
        normalized_advantages = (
            (raw_advantages - raw_advantages.mean())
            / raw_advantages.std(unbiased=False)
        )
    else:
        normalized_advantages = raw_advantages
    advantage_by_transition = {
        id(transition): normalized_advantages[index]
        for index, transition in enumerate(transitions)
    }

    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropy_values: list[float] = []

    for _ in range(ppo_epochs):
        shuffled = list(transitions)
        random.shuffle(shuffled)
        for transition in shuffled:
            logits, value = v6.forward_state(
                network=network,
                current=transition.current,
                target=transition.target,
                actions=transition.actions,
                prior_alpha=prior_alpha,
                device=device,
            )
            penalty_tensor = torch.tensor(
                transition.action_penalties,
                dtype=torch.float32,
                device=device,
            )
            distribution = torch.distributions.Categorical(logits=logits + penalty_tensor)
            action_index = torch.tensor(
                transition.action_index, dtype=torch.long, device=device
            )
            old_log_prob = torch.tensor(
                transition.old_log_prob, dtype=torch.float32, device=device
            )
            advantage = advantage_by_transition[id(transition)]
            ret = torch.tensor(transition.ret, dtype=torch.float32, device=device)

            ratio = torch.exp(distribution.log_prob(action_index) - old_log_prob)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
            policy_loss = -torch.min(unclipped, clipped)
            value_loss = F.smooth_l1_loss(value.squeeze(), ret)
            entropy = distribution.entropy()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropy_values.append(float(entropy.detach().cpu()))

    return {
        "ppo_loss": sum(losses) / len(losses),
        "policy_loss": sum(policy_losses) / len(policy_losses),
        "value_loss": sum(value_losses) / len(value_losses),
        "entropy": sum(entropy_values) / len(entropy_values),
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
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
            name=args.wandb_run_name or "ppo_gae",
            config={k: v for k, v in vars(args).items()
                    if not k.startswith("wandb")},
        )
    elif args.wandb_project and not _WANDB_AVAILABLE:
        print("WARNING: wandb not installed — logging disabled.", flush=True)

    print(json.dumps({
        "device": str(device), "algo": "ppo_gae",
        "gae_lambda": args.gae_lambda, "clip_eps": args.clip_eps,
        "prior_alpha": args.prior_alpha, "wandb": use_wandb,
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
        transitions, rollout_metrics = collect_rollouts_gae(
            env=env,
            network=network,
            examples=examples,
            max_steps=args.max_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            prior_alpha=args.prior_alpha,
            device=device,
        )
        ppo_metrics = update_ppo(
            network=network,
            optimizer=optimizer,
            transitions=transitions,
            prior_alpha=args.prior_alpha,
            clip_eps=args.clip_eps,
            ppo_epochs=args.ppo_epochs,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            normalize_advantages=args.normalize_advantages,
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
        record = {"epoch": epoch, **rollout_metrics, **ppo_metrics, **eval_metrics}
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
