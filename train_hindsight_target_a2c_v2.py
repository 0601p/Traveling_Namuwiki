from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from environment import NamuwikiEnvironment
from evaluate_paths import PathExample, iter_path_examples, limited
from models.neural_target_a2c import NeuralTargetActorCritic, save_checkpoint
from utils import ACTIONS_DATASET, PATHS_DATASET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HindsightTargetA2CV2 with A2C plus HER-style relabeling."
    )
    parser.add_argument("--actions-path", default=ACTIONS_DATASET)
    parser.add_argument("--paths-path", default=PATHS_DATASET)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.92)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--her-k", type=int, default=2)
    parser.add_argument("--her-coef", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bucket-size", type=int, default=50000)
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("checkpoints") / "hindsight_target_a2c_v2.pt",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class EpisodeTransition:
    current: str
    actions: list[str]
    action_index: int
    next_title: str


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"device": str(device)}), flush=True)

    env = NamuwikiEnvironment.from_dataset(args.actions_path)
    examples = list(
        limited(iter_path_examples(args.paths_path, args.train_split), args.train_limit)
    )
    eval_examples = list(
        limited(iter_path_examples(args.paths_path, args.eval_split), args.eval_limit)
    )
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
            device=device,
        )
        eval_metrics = evaluate_policy(
            env=env,
            network=network,
            examples=eval_examples,
            max_steps=args.max_steps,
            device=device,
        )
        record = {"epoch": epoch, **train_metrics, **eval_metrics}
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


def train_epoch(
    *,
    env: NamuwikiEnvironment,
    network: NeuralTargetActorCritic,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[PathExample],
    max_steps: int,
    gamma: float,
    entropy_coef: float,
    value_coef: float,
    her_k: int,
    her_coef: float,
    device: torch.device,
) -> dict:
    network.train()
    losses: list[float] = []
    her_losses: list[float] = []
    returns: list[float] = []
    successes = 0

    for example in examples:
        current = example.start_title
        seen = {current}
        episode_return = 0.0
        transitions: list[EpisodeTransition] = []
        achieved_titles: list[str] = []

        for _ in range(max_steps):
            actions = list(env.actions(current))
            if not actions:
                episode_return += -0.2
                break

            logits, value = forward_state(
                network=network,
                current=current,
                target=example.target_title,
                actions=actions,
                device=device,
            )
            penalties = torch.tensor(
                [-0.5 if action in seen else 0.0 for action in actions],
                dtype=torch.float32,
                device=device,
            )
            distribution = torch.distributions.Categorical(logits=logits + penalties)
            action_index = distribution.sample()
            chosen_index = int(action_index.item())
            action = actions[chosen_index]
            next_seen = action in seen
            reached_target = action == example.target_title
            dead_end = not env.actions(action)

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
                    _, next_value_tensor = forward_state(
                        network=network,
                        current=action,
                        target=example.target_title,
                        actions=list(env.actions(action)) or [action],
                        device=device,
                    )
                    next_value = next_value_tensor.squeeze()
                target_value = torch.tensor(reward, dtype=torch.float32, device=device)
                target_value = target_value + gamma * next_value

            advantage = target_value - value.squeeze()
            policy_loss = -distribution.log_prob(action_index) * advantage.detach()
            value_loss = F.smooth_l1_loss(value.squeeze(), target_value)
            entropy_loss = -distribution.entropy()
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            episode_return += reward
            transitions.append(
                EpisodeTransition(current, actions, chosen_index, action)
            )
            achieved_titles.append(action)
            if reached_target:
                successes += 1
            if done:
                break

            seen.add(action)
            current = action

        if transitions:
            her_losses.extend(
                apply_hindsight_updates(
                    network=network,
                    optimizer=optimizer,
                    transitions=transitions,
                    achieved_titles=achieved_titles,
                    gamma=gamma,
                    her_k=her_k,
                    her_coef=her_coef,
                    device=device,
                )
            )
        returns.append(episode_return)

    return {
        "train_loss": sum(losses) / len(losses) if losses else 0.0,
        "hindsight_loss": sum(her_losses) / len(her_losses) if her_losses else 0.0,
        "avg_return": sum(returns) / len(returns),
        "train_success_rate": successes / len(examples),
    }


def apply_hindsight_updates(
    *,
    network: NeuralTargetActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[EpisodeTransition],
    achieved_titles: Sequence[str],
    gamma: float,
    her_k: int,
    her_coef: float,
    device: torch.device,
) -> list[float]:
    losses: list[float] = []
    if her_k <= 0:
        return losses

    for transition_index, transition in enumerate(transitions):
        future_indexes = list(range(transition_index, len(achieved_titles)))
        sampled_indexes = random.sample(
            future_indexes,
            k=min(her_k, len(future_indexes)),
        )
        for future_index in sampled_indexes:
            hindsight_goal = achieved_titles[future_index]
            distance_to_goal = future_index - transition_index + 1
            logits, value = forward_state(
                network=network,
                current=transition.current,
                target=hindsight_goal,
                actions=transition.actions,
                device=device,
            )
            target_action = torch.tensor(
                [transition.action_index],
                dtype=torch.long,
                device=device,
            )
            actor_loss = F.cross_entropy(logits.unsqueeze(0), target_action)
            target_value = torch.tensor(
                gamma ** (distance_to_goal - 1),
                dtype=torch.float32,
                device=device,
            )
            critic_loss = F.smooth_l1_loss(value.squeeze(), target_value)
            loss = her_coef * (actor_loss + 0.5 * critic_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
    return losses


def forward_state(
    *,
    network: NeuralTargetActorCritic,
    current: str,
    target: str,
    actions: Sequence[str],
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
    value = network.value(current_embedding, target_embedding)
    return logits, value


@torch.no_grad()
def evaluate_policy(
    *,
    env: NamuwikiEnvironment,
    network: NeuralTargetActorCritic,
    examples: Sequence[PathExample],
    max_steps: int,
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
