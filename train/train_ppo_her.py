"""
train_ppo_her.py — PPO with GAE plus HER relabeling.

Each epoch:
  1. Roll out the current policy on start/target examples.
  2. Update the original goal-conditioned policy with PPO/GAE.
  3. Reuse the same trajectories with hindsight goals as an auxiliary HER loss.
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
from evaluate_paths import PathExample, iter_path_examples, limited
from models.networks.neural_target_actor_critic import NeuralTargetActorCritic, save_checkpoint
from similarity import cached_similarity
from utils import ACTIONS_DATASET, PATHS_DATASET

import train_greedy_residual_ppo_v6 as v6
import train_hindsight_target_a2c_v2 as base
import train_her_variation as her
import train_ppo as ppo

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


@dataclass(frozen=True)
class EpisodeRollout:
    target: str
    ppo_transitions: list[ppo.Transition]
    her_transitions: list[base.EpisodeTransition]
    achieved_titles: list[str]
    episode_return: float
    reached_target: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO with GAE and HER goal relabeling.",
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
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument(
        "--no-normalize-advantages",
        dest="normalize_advantages",
        action="store_false",
        help="Disable PPO advantage normalization.",
    )
    parser.add_argument(
        "--her-strategy",
        choices=["future", "final", "all", "semantic", "mixed"],
        default="mixed",
    )
    parser.add_argument("--her-k", type=int, default=2)
    parser.add_argument("--her-coef", type=float, default=0.1)
    parser.add_argument("--her-temp", type=float, default=1.0)
    parser.add_argument(
        "--bc-aux-coef",
        type=float,
        default=0.0,
        help="Auxiliary behavioral-cloning loss weight applied each epoch.",
    )
    parser.add_argument(
        "--bc-aux-samples",
        type=int,
        default=1024,
        help="Number of BC samples used per epoch when bc_aux_coef > 0.",
    )
    parser.add_argument("--bc-samples-mult", type=int, default=4)
    parser.add_argument(
        "--shaping-beta",
        type=float,
        default=0.0,
        help="Potential-based reward shaping strength using title lexical similarity.",
    )
    parser.add_argument("--prior-alpha", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bucket-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("checkpoints") / "ppo_her.pt",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint used to warm-start the actor-critic network.",
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    return parser.parse_args()


def collect_rollouts_gae_with_her(
    *,
    env: NamuwikiEnvironment,
    network: NeuralTargetActorCritic,
    examples: Sequence[PathExample],
    max_steps: int,
    gamma: float,
    gae_lambda: float,
    prior_alpha: float,
    shaping_beta: float,
    device: torch.device,
) -> tuple[list[ppo.Transition], list[EpisodeRollout], dict]:
    network.eval()
    all_transitions: list[ppo.Transition] = []
    episodes: list[EpisodeRollout] = []
    successes = 0
    returns: list[float] = []
    base_returns: list[float] = []

    for example in examples:
        episode_steps: list[dict] = []
        her_transitions: list[base.EpisodeTransition] = []
        achieved_titles: list[str] = []
        episode_return = 0.0
        episode_base_return = 0.0
        current = example.start_title
        seen = {current}
        reached_episode_target = False

        for _ in range(max_steps):
            actions = list(env.actions(current))
            if not actions:
                episode_return += -0.2
                episode_base_return += -0.2
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
            chosen_index = int(action_index.item())
            action = actions[chosen_index]

            reached_target = action == example.target_title
            next_seen = action in seen
            dead_end = not list(env.actions(action))
            done = reached_target or next_seen or dead_end

            base_reward = -0.05
            if reached_target:
                base_reward = 1.0
                successes += 1
                reached_episode_target = True
            elif next_seen:
                base_reward = -0.35
            elif dead_end:
                base_reward = -0.2

            reward = base_reward
            if shaping_beta:
                current_phi = cached_similarity(current, example.target_title)
                next_phi = cached_similarity(action, example.target_title)
                reward += shaping_beta * (gamma * next_phi - current_phi)

            episode_steps.append(
                {
                    "current": current,
                    "target": example.target_title,
                    "actions": actions,
                    "action_penalties": action_penalties,
                    "action_index": chosen_index,
                    "old_log_prob": float(dist.log_prob(action_index).detach().cpu()),
                    "value": float(value.squeeze().detach().cpu()),
                    "reward": reward,
                    "base_reward": base_reward,
                    "done": done,
                }
            )
            her_transitions.append(
                base.EpisodeTransition(current, actions, chosen_index, action)
            )
            achieved_titles.append(action)
            episode_return += reward
            episode_base_return += base_reward

            if done:
                break
            seen.add(action)
            current = action

        next_value = 0.0
        gae = 0.0
        for step in reversed(episode_steps):
            bootstrap_value = 0.0 if step["done"] else next_value
            delta = step["reward"] + gamma * bootstrap_value - step["value"]
            gae = delta + gamma * gae_lambda * (0.0 if step["done"] else gae)
            step["advantage"] = gae
            step["ret"] = gae + step["value"]
            next_value = step["value"]

        ppo_transitions: list[ppo.Transition] = []
        for step in episode_steps:
            transition = ppo.Transition(
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
            )
            ppo_transitions.append(transition)
            all_transitions.append(transition)

        if ppo_transitions:
            episodes.append(
                EpisodeRollout(
                    target=example.target_title,
                    ppo_transitions=ppo_transitions,
                    her_transitions=her_transitions,
                    achieved_titles=achieved_titles,
                    episode_return=episode_return,
                    reached_target=reached_episode_target,
                )
            )
        returns.append(episode_return)
        base_returns.append(episode_base_return)

    return all_transitions, episodes, {
        "rollout_return": sum(returns) / len(returns) if returns else 0.0,
        "base_rollout_return": (
            sum(base_returns) / len(base_returns) if base_returns else 0.0
        ),
        "train_success_rate": successes / len(examples) if examples else 0.0,
        "transitions": len(all_transitions),
        "episodes_with_actions": len(episodes),
    }


def apply_her_auxiliary_updates(
    *,
    network: NeuralTargetActorCritic,
    optimizer: torch.optim.Optimizer,
    episodes: Sequence[EpisodeRollout],
    gamma: float,
    her_k: int,
    her_coef: float,
    her_strategy: str,
    her_temp: float,
    device: torch.device,
) -> dict:
    losses: list[float] = []
    update_count = 0
    for episode in episodes:
        if not episode.her_transitions:
            continue
        episode_losses, episode_updates = her.apply_hindsight_updates_varied(
            network=network,
            optimizer=optimizer,
            transitions=episode.her_transitions,
            achieved_titles=episode.achieved_titles,
            original_target=episode.target,
            gamma=gamma,
            her_k=her_k,
            her_coef=her_coef,
            her_strategy=her_strategy,
            her_temp=her_temp,
            device=device,
        )
        losses.extend(episode_losses)
        update_count += episode_updates

    return {
        "hindsight_loss": sum(losses) / len(losses) if losses else 0.0,
        "hindsight_updates": update_count,
    }


def load_initial_checkpoint(
    network: NeuralTargetActorCritic,
    path: Path | None,
    *,
    device: torch.device,
) -> NeuralTargetActorCritic:
    if path is None:
        return network
    if not path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {path}")

    payload = torch.load(path, map_location=device)
    config = payload.get("config", {})
    if config:
        network = NeuralTargetActorCritic(**config).to(device)
    state_dict = payload.get("state_dict", payload)
    network.load_state_dict(state_dict)
    network.train()
    return network


def apply_bc_auxiliary_updates(
    *,
    network: NeuralTargetActorCritic,
    optimizer: torch.optim.Optimizer,
    samples: Sequence[v6.BCSample],
    sample_count: int,
    bc_aux_coef: float,
    prior_alpha: float,
    device: torch.device,
) -> dict:
    if bc_aux_coef <= 0.0 or sample_count <= 0 or not samples:
        return {"bc_aux_loss": 0.0, "bc_aux_acc": 0.0, "bc_aux_samples": 0}

    network.train()
    selected = random.sample(list(samples), k=min(sample_count, len(samples)))
    losses: list[float] = []
    correct = 0
    for sample in selected:
        logits, _ = v6.forward_state(
            network=network,
            current=sample.current,
            target=sample.target,
            actions=sample.actions,
            prior_alpha=prior_alpha,
            device=device,
        )
        label = torch.tensor([sample.action_index], dtype=torch.long, device=device)
        ce_loss = F.cross_entropy(logits.unsqueeze(0), label)
        loss = bc_aux_coef * ce_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()

        losses.append(float(ce_loss.detach().cpu()))
        correct += int(torch.argmax(logits).item() == sample.action_index)

    return {
        "bc_aux_loss": sum(losses) / len(losses) if losses else 0.0,
        "bc_aux_acc": correct / len(selected) if selected else 0.0,
        "bc_aux_samples": len(selected),
    }


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
            name=args.wandb_run_name or f"ppo_her_{args.her_strategy}",
            config={k: v for k, v in vars(args).items() if not k.startswith("wandb")},
        )
    elif args.wandb_project and not _WANDB_AVAILABLE:
        print("WARNING: wandb not installed — logging disabled.", flush=True)

    print(
        json.dumps(
            {
                "device": str(device),
                "algo": "ppo_her",
                "gae_lambda": args.gae_lambda,
                "clip_eps": args.clip_eps,
                "her_strategy": args.her_strategy,
                "her_k": args.her_k,
                "her_coef": args.her_coef,
                "bc_aux_coef": args.bc_aux_coef,
                "shaping_beta": args.shaping_beta,
                "prior_alpha": args.prior_alpha,
                "wandb": use_wandb,
            }
        ),
        flush=True,
    )

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
    network = load_initial_checkpoint(
        network,
        args.init_checkpoint,
        device=device,
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)

    bc_aux_samples: list[v6.BCSample] = []
    if args.bc_aux_coef > 0.0 and args.bc_aux_samples > 0:
        bc_aux_samples = list(
            limited(
                v6.iter_bc_samples(
                    env=env,
                    dataset_path=args.paths_path,
                    split=args.train_split,
                ),
                args.train_limit * args.bc_samples_mult,
            )
        )

    best_success = -1.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        transitions, episodes, rollout_metrics = collect_rollouts_gae_with_her(
            env=env,
            network=network,
            examples=examples,
            max_steps=args.max_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            prior_alpha=args.prior_alpha,
            shaping_beta=args.shaping_beta,
            device=device,
        )
        ppo_metrics = ppo.update_ppo(
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
        her_metrics = apply_her_auxiliary_updates(
            network=network,
            optimizer=optimizer,
            episodes=episodes,
            gamma=args.gamma,
            her_k=args.her_k,
            her_coef=args.her_coef,
            her_strategy=args.her_strategy,
            her_temp=args.her_temp,
            device=device,
        )
        bc_aux_metrics = apply_bc_auxiliary_updates(
            network=network,
            optimizer=optimizer,
            samples=bc_aux_samples,
            sample_count=args.bc_aux_samples,
            bc_aux_coef=args.bc_aux_coef,
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
        record = {
            "epoch": epoch,
            "her_strategy": args.her_strategy,
            **rollout_metrics,
            **ppo_metrics,
            **her_metrics,
            **bc_aux_metrics,
            **eval_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

        if use_wandb:
            _wandb.log(
                {k: v for k, v in record.items() if isinstance(v, (int, float))},
                step=epoch,
            )

        if eval_metrics["success_rate"] > best_success:
            best_success = eval_metrics["success_rate"]
            save_checkpoint(path=args.checkpoint_output, network=network, metrics=record)
            if use_wandb:
                _wandb.summary["best_success_rate"] = best_success
                _wandb.summary["best_epoch"] = epoch

    summary = {
        "checkpoint": str(args.checkpoint_output),
        "best_success_rate": best_success,
        "her_strategy": args.her_strategy,
        "history": history,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if use_wandb:
        _wandb.finish()


if __name__ == "__main__":
    main()
