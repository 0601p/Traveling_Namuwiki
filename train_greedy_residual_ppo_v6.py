from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from dataset_utils import load_rows
from environment import NamuwikiEnvironment
from evaluate_paths import PathExample, iter_path_examples, limited
from models.neural_target_a2c import NeuralTargetActorCritic, save_checkpoint
from utils import ACTIONS_DATASET, PATHS_DATASET, cached_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lexical-greedy residual PPO v6.")
    parser.add_argument(
        "--algo",
        choices=["residual", "bc", "sil", "bc_sil"],
        default="bc_sil",
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--bc-epochs", type=int, default=1)
    parser.add_argument("--bc-samples-mult", type=int, default=4)
    parser.add_argument("--sil-coef", type=float, default=0.25)
    parser.add_argument("--sil-buffer-size", type=int, default=4000)
    parser.add_argument("--prior-alpha", type=float, default=5.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bucket-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("checkpoints") / "greedy_residual_ppo_v6.pt",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class Transition:
    current: str
    target: str
    actions: list[str]
    action_index: int
    old_log_prob: float
    value: float
    reward: float
    ret: float
    advantage: float


@dataclass(frozen=True)
class BCSample:
    current: str
    target: str
    actions: list[str]
    action_index: int


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "device": str(device),
                "algo": args.algo,
                "prior_alpha": args.prior_alpha,
                "lr": args.lr,
                "clip_eps": args.clip_eps,
            }
        ),
        flush=True,
    )

    env = NamuwikiEnvironment.from_dataset(args.actions_path)
    examples = list(limited(iter_path_examples(args.paths_path, args.train_split), args.train_limit))
    eval_examples = list(limited(iter_path_examples(args.paths_path, args.eval_split), args.eval_limit))
    network = NeuralTargetActorCritic(
        bucket_size=args.bucket_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)

    if args.algo in {"bc", "bc_sil"}:
        bc_samples = list(
            limited(
                iter_bc_samples(env=env, dataset_path=args.paths_path, split=args.train_split),
                args.train_limit * args.bc_samples_mult,
            )
        )
        run_bc_pretrain(
            network=network,
            optimizer=optimizer,
            samples=bc_samples,
            prior_alpha=args.prior_alpha,
            epochs=args.bc_epochs,
            device=device,
        )

    best_success = -1.0
    history: list[dict] = []
    sil_buffer: list[Transition] = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        transitions, rollout_metrics = collect_rollouts(
            env=env,
            network=network,
            examples=examples,
            max_steps=args.max_steps,
            gamma=args.gamma,
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
            device=device,
        )
        sil_metrics = {"sil_loss": 0.0, "sil_updates": 0}
        if args.algo in {"sil", "bc_sil"}:
            sil_buffer.extend([transition for transition in transitions if transition.ret > 0.5])
            sil_buffer = sil_buffer[-args.sil_buffer_size :]
            sil_metrics = update_sil(
                network=network,
                optimizer=optimizer,
                transitions=sil_buffer,
                prior_alpha=args.prior_alpha,
                sil_coef=args.sil_coef,
                device=device,
            )
        eval_metrics = evaluate_policy(
            env=env,
            network=network,
            examples=eval_examples,
            max_steps=args.max_steps,
            prior_alpha=args.prior_alpha,
            device=device,
        )
        record = {"epoch": epoch, **rollout_metrics, **ppo_metrics, **sil_metrics, **eval_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if eval_metrics["success_rate"] > best_success:
            best_success = eval_metrics["success_rate"]
            save_checkpoint(path=args.checkpoint_output, network=network, metrics=record)

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


def iter_bc_samples(
    *,
    env: NamuwikiEnvironment,
    dataset_path: str,
    split: str,
) -> Iterable[BCSample]:
    for row in load_rows(dataset_path, split=split):
        start = str(row.get("start_title") or "").strip()
        target = str(row.get("target_title") or "").strip()
        if not start or not target:
            continue
        for path in row.get("paths", []):
            full_path = [start, *[str(title) for title in path], target]
            for current, next_title in zip(full_path, full_path[1:]):
                actions = list(env.actions(current))
                if next_title in actions:
                    yield BCSample(current, target, actions, actions.index(next_title))


def run_bc_pretrain(
    *,
    network: NeuralTargetActorCritic,
    optimizer: torch.optim.Optimizer,
    samples: Sequence[BCSample],
    prior_alpha: float,
    epochs: int,
    device: torch.device,
) -> None:
    network.train()
    for epoch in range(1, epochs + 1):
        shuffled = list(samples)
        random.shuffle(shuffled)
        losses: list[float] = []
        correct = 0
        for sample in shuffled:
            logits, _ = forward_state(
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
        print(
            json.dumps(
                {
                    "bc_epoch": epoch,
                    "bc_samples": len(samples),
                    "bc_loss": sum(losses) / len(losses) if losses else 0.0,
                    "bc_acc": correct / len(samples) if samples else 0.0,
                }
            ),
            flush=True,
        )


def collect_rollouts(
    *,
    env: NamuwikiEnvironment,
    network: NeuralTargetActorCritic,
    examples: Sequence[PathExample],
    max_steps: int,
    gamma: float,
    prior_alpha: float,
    device: torch.device,
) -> tuple[list[Transition], dict]:
    network.eval()
    transitions: list[Transition] = []
    successes = 0
    returns: list[float] = []
    for example in examples:
        episode_steps: list[dict] = []
        current = example.start_title
        seen = {current}
        for _ in range(max_steps):
            actions = list(env.actions(current))
            if not actions:
                break
            logits, value = forward_state(
                network=network,
                current=current,
                target=example.target_title,
                actions=actions,
                prior_alpha=prior_alpha,
                device=device,
            )
            penalties = torch.tensor(
                [-0.5 if action in seen else 0.0 for action in actions],
                dtype=torch.float32,
                device=device,
            )
            distribution = torch.distributions.Categorical(logits=logits + penalties)
            action_index = distribution.sample()
            action = actions[int(action_index.item())]
            reached_target = action == example.target_title
            next_seen = action in seen
            dead_end = not env.actions(action)
            reward = -0.05
            if reached_target:
                reward = 1.0
            elif next_seen:
                reward = -0.35
            elif dead_end:
                reward = -0.2
            episode_steps.append(
                {
                    "current": current,
                    "target": example.target_title,
                    "actions": actions,
                    "action_index": int(action_index.item()),
                    "old_log_prob": float(distribution.log_prob(action_index).detach().cpu()),
                    "value": float(value.squeeze().detach().cpu()),
                    "reward": reward,
                }
            )
            if reached_target:
                successes += 1
            if reached_target or next_seen or dead_end:
                break
            seen.add(action)
            current = action

        ret = 0.0
        episode_transitions: list[Transition] = []
        for step in reversed(episode_steps):
            ret = step["reward"] + gamma * ret
            episode_transitions.append(
                Transition(
                    current=step["current"],
                    target=step["target"],
                    actions=step["actions"],
                    action_index=step["action_index"],
                    old_log_prob=step["old_log_prob"],
                    value=step["value"],
                    reward=step["reward"],
                    ret=ret,
                    advantage=ret - step["value"],
                )
            )
        episode_transitions.reverse()
        transitions.extend(episode_transitions)
        returns.append(sum(step["reward"] for step in episode_steps))
    return transitions, {
        "rollout_return": sum(returns) / len(returns) if returns else 0.0,
        "train_success_rate": successes / len(examples) if examples else 0.0,
        "transitions": len(transitions),
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
    device: torch.device,
) -> dict:
    network.train()
    if not transitions:
        return {"ppo_loss": 0.0}
    losses: list[float] = []
    for _ in range(ppo_epochs):
        shuffled = list(transitions)
        random.shuffle(shuffled)
        for transition in shuffled:
            logits, value = forward_state(
                network=network,
                current=transition.current,
                target=transition.target,
                actions=transition.actions,
                prior_alpha=prior_alpha,
                device=device,
            )
            distribution = torch.distributions.Categorical(logits=logits)
            action_index = torch.tensor(transition.action_index, dtype=torch.long, device=device)
            old_log_prob = torch.tensor(transition.old_log_prob, dtype=torch.float32, device=device)
            advantage = torch.tensor(transition.advantage, dtype=torch.float32, device=device)
            ret = torch.tensor(transition.ret, dtype=torch.float32, device=device)
            ratio = torch.exp(distribution.log_prob(action_index) - old_log_prob)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
            policy_loss = -torch.min(unclipped, clipped)
            value_loss = F.smooth_l1_loss(value.squeeze(), ret)
            entropy_loss = -distribution.entropy()
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {"ppo_loss": sum(losses) / len(losses)}


def update_sil(
    *,
    network: NeuralTargetActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    prior_alpha: float,
    sil_coef: float,
    device: torch.device,
) -> dict:
    network.train()
    if not transitions:
        return {"sil_loss": 0.0, "sil_updates": 0}
    sampled = random.sample(list(transitions), k=min(500, len(transitions)))
    losses: list[float] = []
    for transition in sampled:
        logits, value = forward_state(
            network=network,
            current=transition.current,
            target=transition.target,
            actions=transition.actions,
            prior_alpha=prior_alpha,
            device=device,
        )
        distribution = torch.distributions.Categorical(logits=logits)
        action_index = torch.tensor(transition.action_index, dtype=torch.long, device=device)
        ret = torch.tensor(transition.ret, dtype=torch.float32, device=device)
        advantage = torch.clamp(ret - value.squeeze(), min=0.0)
        policy_loss = -distribution.log_prob(action_index) * advantage.detach()
        value_loss = F.smooth_l1_loss(value.squeeze(), ret)
        loss = sil_coef * (policy_loss + 0.5 * value_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"sil_loss": sum(losses) / len(losses), "sil_updates": len(sampled)}


def forward_state(
    *,
    network: NeuralTargetActorCritic,
    current: str,
    target: str,
    actions: Sequence[str],
    prior_alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = [current, target, *actions]
    embeddings = network.encode_texts(texts, device=device)
    current_embedding = embeddings[0:1]
    target_embedding = embeddings[1:2]
    action_embeddings = embeddings[2:]
    logits = network.actor_logits(
        current_embedding=current_embedding,
        target_embedding=target_embedding,
        action_embeddings=action_embeddings,
    )
    if prior_alpha:
        prior = torch.tensor(
            [cached_similarity(action, target) for action in actions],
            dtype=torch.float32,
            device=device,
        )
        logits = logits + prior_alpha * prior
    value = network.value(current_embedding, target_embedding)
    return logits, value


@torch.no_grad()
def evaluate_policy(
    *,
    env: NamuwikiEnvironment,
    network: NeuralTargetActorCritic,
    examples: Sequence[PathExample],
    max_steps: int,
    prior_alpha: float,
    device: torch.device,
) -> dict:
    network.eval()
    successes = 0
    success_steps: list[int] = []
    for example in examples:
        current = example.start_title
        seen = {current}
        for step in range(1, max_steps + 1):
            actions = list(env.actions(current))
            if not actions:
                break
            logits, _ = forward_state(
                network=network,
                current=current,
                target=example.target_title,
                actions=actions,
                prior_alpha=prior_alpha,
                device=device,
            )
            penalties = torch.tensor(
                [-0.5 if action in seen else 0.0 for action in actions],
                dtype=torch.float32,
                device=device,
            )
            action = actions[int(torch.argmax(logits + penalties).item())]
            if action == example.target_title:
                successes += 1
                success_steps.append(step)
                break
            if action in seen:
                break
            seen.add(action)
            current = action
    return {
        "success_rate": successes / len(examples) if examples else 0.0,
        "mean_steps_on_success": (
            sum(success_steps) / len(success_steps) if success_steps else None
        ),
        "count": len(examples),
    }


if __name__ == "__main__":
    main()
