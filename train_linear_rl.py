from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import torch
from datasets import load_dataset

from environment import NamuwikiEnvironment
from models.linear import load_embeddings, load_weights
from utils import ACTIONS_DATASET, PATHS_DATASET, Title


@dataclass(frozen=True)
class PathExample:
    start_title: Title
    target_title: Title
    hop: int


@dataclass
class Episode:
    log_probs: list[torch.Tensor]
    entropies: list[torch.Tensor]
    reward: float
    reached_target: bool
    steps_taken: int
    stopped_reason: str


class LinearPolicy(torch.nn.Module):
    def __init__(
        self,
        embeddings: dict[str, list[float]],
        *,
        weights_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        first_key = next(iter(embeddings))
        self.embedding_dim = len(embeddings[first_key])
        self.device = torch.device(device)
        self.embeddings = {
            title: torch.tensor(vector, dtype=torch.float32, device=self.device)
            for title, vector in embeddings.items()
        }
        (
            link_weights,
            target_weights,
            interaction_weights,
            bias,
        ) = load_weights(weights_path, embedding_dim=self.embedding_dim)
        self.link_weights = torch.nn.Parameter(
            torch.tensor(link_weights, dtype=torch.float32, device=self.device)
        )
        self.target_weights = torch.nn.Parameter(
            torch.tensor(target_weights, dtype=torch.float32, device=self.device)
        )
        self.interaction_weights = torch.nn.Parameter(
            torch.tensor(interaction_weights, dtype=torch.float32, device=self.device)
        )
        self.bias = torch.nn.Parameter(
            torch.tensor(float(bias), dtype=torch.float32, device=self.device)
        )

    def scores_for_actions(
        self,
        actions: list[str],
        target: str,
    ) -> tuple[list[str], torch.Tensor] | None:
        target_embedding = self.embeddings.get(target)
        if target_embedding is None:
            return None

        valid_actions: list[str] = []
        valid_embeddings: list[torch.Tensor] = []
        for action in actions:
            action_embedding = self.embeddings.get(action)
            if action_embedding is None:
                continue
            valid_actions.append(action)
            valid_embeddings.append(action_embedding)

        if not valid_actions:
            return None

        action_matrix = torch.stack(valid_embeddings, dim=0)
        target_matrix = target_embedding.unsqueeze(0).expand_as(action_matrix)
        interaction_matrix = action_matrix * target_matrix
        scores = (
            action_matrix @ self.link_weights
            + target_matrix @ self.target_weights
            + interaction_matrix @ self.interaction_weights
            + self.bias
        )
        return valid_actions, scores

    def sample_action(
        self,
        actions: list[str],
        target: str,
        *,
        temperature: float,
    ) -> tuple[str, torch.Tensor, torch.Tensor] | None:
        scored = self.scores_for_actions(actions, target)
        if scored is None:
            return None
        valid_actions, scores = scored
        scaled_scores = scores / max(temperature, 1e-6)
        distribution = torch.distributions.Categorical(logits=scaled_scores)
        index = distribution.sample()
        return (
            valid_actions[int(index.item())],
            distribution.log_prob(index),
            distribution.entropy(),
        )

    def greedy_action(self, actions: list[str], target: str) -> str | None:
        scored = self.scores_for_actions(actions, target)
        if scored is None:
            return None
        valid_actions, scores = scored
        best_index = int(torch.argmax(scores).item())
        return valid_actions[best_index]

    def export_weights(self) -> dict[str, object]:
        return {
            "link_weights": self.link_weights.detach().cpu().tolist(),
            "target_weights": self.target_weights.detach().cpu().tolist(),
            "interaction_weights": self.interaction_weights.detach().cpu().tolist(),
            "bias": float(self.bias.detach().cpu().item()),
        }


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear link-scoring weights with policy gradient episodes."
    )
    parser.add_argument("--actions-path", default=ACTIONS_DATASET)
    parser.add_argument("--paths-path", default=PATHS_DATASET)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--init-weights-path")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--failure-reward", type=float, default=0.0)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/rl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print a training progress log every N batches.",
    )
    parser.add_argument(
        "--allow-cycle",
        dest="stop_on_cycle",
        action="store_false",
        help="Keep episodes running even if a title is revisited.",
    )
    parser.set_defaults(stop_on_cycle=True)
    return parser.parse_args()


def row_to_example(row: dict) -> PathExample:
    start_title = str(row.get("start_title") or "").strip()
    target_title = str(row.get("target_title") or "").strip()
    if not start_title or not target_title:
        raise ValueError(f"Missing start_title or target_title: {row}")
    return PathExample(
        start_title=start_title,
        target_title=target_title,
        hop=int(row["hop"]),
    )


def load_examples(
    dataset_path: str,
    split: str,
    *,
    limit: int | None = None,
) -> list[PathExample]:
    examples: list[PathExample] = []
    for index, row in enumerate(load_dataset(dataset_path, split=split)):
        if limit is not None and index >= limit:
            break
        examples.append(row_to_example(row))
    return examples


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw_device: str) -> str:
    if raw_device != "auto":
        return raw_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def discounted_returns(length: int, final_reward: float, discount: float) -> list[float]:
    returns = [0.0] * length
    running = final_reward
    for index in range(length - 1, -1, -1):
        returns[index] = running
        running *= discount
    return returns


def run_episode(
    env: NamuwikiEnvironment,
    policy: LinearPolicy,
    example: PathExample,
    *,
    max_steps: int,
    stop_on_cycle: bool,
    temperature: float,
    step_penalty: float,
    success_reward: float,
    failure_reward: float,
) -> Episode:
    current = example.start_title
    seen = {current}
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []

    for step_index in range(max_steps):
        actions = list(env.actions(current))
        sampled = policy.sample_action(
            actions,
            example.target_title,
            temperature=temperature,
        )
        if sampled is None:
            reward = failure_reward - step_penalty * step_index
            return Episode(log_probs, entropies, reward, False, step_index, "no_actions")

        next_title, log_prob, entropy = sampled
        log_probs.append(log_prob)
        entropies.append(entropy)

        if next_title == example.target_title:
            reward = success_reward - step_penalty * (step_index + 1)
            return Episode(
                log_probs,
                entropies,
                reward,
                True,
                step_index + 1,
                "target_reached",
            )

        if stop_on_cycle and next_title in seen:
            reward = failure_reward - step_penalty * (step_index + 1)
            return Episode(log_probs, entropies, reward, False, step_index + 1, "cycle")

        seen.add(next_title)
        current = next_title

    reward = failure_reward - step_penalty * max_steps
    return Episode(log_probs, entropies, reward, False, max_steps, "max_steps")


def train_batch(
    env: NamuwikiEnvironment,
    policy: LinearPolicy,
    optimizer: torch.optim.Optimizer,
    batch: list[PathExample],
    *,
    args: argparse.Namespace,
) -> dict[str, float]:
    episodes = [
        run_episode(
            env,
            policy,
            example,
            max_steps=args.max_steps,
            stop_on_cycle=args.stop_on_cycle,
            temperature=args.temperature,
            step_penalty=args.step_penalty,
            success_reward=args.success_reward,
            failure_reward=args.failure_reward,
        )
        for example in batch
    ]

    all_returns: list[float] = []
    log_prob_terms: list[torch.Tensor] = []
    entropy_terms: list[torch.Tensor] = []

    for episode in episodes:
        if not episode.log_probs:
            continue
        returns = discounted_returns(
            len(episode.log_probs),
            episode.reward,
            args.discount,
        )
        all_returns.extend(returns)
        log_prob_terms.extend(episode.log_probs)
        entropy_terms.extend(episode.entropies)

    if not log_prob_terms:
        return {
            "loss": 0.0,
            "avg_reward": sum(ep.reward for ep in episodes) / max(len(episodes), 1),
            "success_rate": sum(ep.reached_target for ep in episodes) / max(len(episodes), 1),
        }

    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=policy.device)
    centered_returns = returns_tensor - returns_tensor.mean()
    if len(centered_returns) > 1:
        std = centered_returns.std(unbiased=False)
        if std.item() > 0:
            centered_returns = centered_returns / std

    log_prob_tensor = torch.stack(log_prob_terms)
    entropy_tensor = torch.stack(entropy_terms)
    loss = -(log_prob_tensor * centered_returns).mean()
    loss -= args.entropy_coef * entropy_tensor.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "avg_reward": sum(ep.reward for ep in episodes) / len(episodes),
        "success_rate": sum(ep.reached_target for ep in episodes) / len(episodes),
    }


def evaluate_policy(
    env: NamuwikiEnvironment,
    policy: LinearPolicy,
    examples: Iterable[PathExample],
    *,
    max_steps: int,
    stop_on_cycle: bool,
) -> dict[str, float]:
    total = 0
    success = 0
    total_steps = 0

    for example in examples:
        total += 1
        current = example.start_title
        seen = {current}
        steps_taken = 0

        for step_index in range(max_steps):
            action = policy.greedy_action(list(env.actions(current)), example.target_title)
            if action is None:
                break
            steps_taken = step_index + 1
            if action == example.target_title:
                success += 1
                break
            if stop_on_cycle and action in seen:
                break
            seen.add(action)
            current = action

        total_steps += steps_taken

    if total == 0:
        return {"success_rate": 0.0, "avg_steps": 0.0}
    return {
        "success_rate": success / total,
        "avg_steps": total_steps / total,
    }


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    log(
        "[train_linear_rl] "
        f"device={device} torch_cuda_available={torch.cuda.is_available()} "
        f"seed={args.seed} epochs={args.epochs} batch_size={args.batch_size}"
    )

    log("[train_linear_rl] loading environment")
    env = NamuwikiEnvironment.from_dataset(args.actions_path)
    log("[train_linear_rl] loading embeddings")
    embeddings = load_embeddings(args.embeddings_path)
    log(f"[train_linear_rl] loaded {len(embeddings)} embeddings")
    policy = LinearPolicy(
        embeddings,
        weights_path=args.init_weights_path,
        device=device,
    )
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    log("[train_linear_rl] loading train/eval examples")
    train_examples = load_examples(args.paths_path, args.train_split, limit=args.train_limit)
    eval_examples = load_examples(args.paths_path, args.eval_split, limit=args.eval_limit)
    if not train_examples:
        raise ValueError("No training examples loaded")
    log(
        "[train_linear_rl] "
        f"train_examples={len(train_examples)} eval_examples={len(eval_examples)}"
    )

    history: list[dict[str, float | int]] = []
    best_success_rate = -math.inf
    best_weights_path = args.save_dir / "best_weights.json"

    for epoch in range(1, args.epochs + 1):
        log(f"[train_linear_rl] starting epoch {epoch}/{args.epochs}")
        shuffled = train_examples[:]
        random.shuffle(shuffled)

        batch_metrics: list[dict[str, float]] = []
        total_batches = (len(shuffled) + args.batch_size - 1) // args.batch_size
        for batch_index, start in enumerate(range(0, len(shuffled), args.batch_size), start=1):
            batch = shuffled[start : start + args.batch_size]
            metrics = train_batch(env, policy, optimizer, batch, args=args)
            batch_metrics.append(metrics)
            if (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % max(args.log_every, 1) == 0
            ):
                log(
                    "[train_linear_rl] "
                    f"epoch {epoch}/{args.epochs} "
                    f"batch {batch_index}/{total_batches} "
                    f"loss={metrics['loss']:.4f} "
                    f"avg_reward={metrics['avg_reward']:.4f} "
                    f"success_rate={metrics['success_rate']:.4f}"
                )

        train_loss = sum(item["loss"] for item in batch_metrics) / len(batch_metrics)
        train_reward = sum(item["avg_reward"] for item in batch_metrics) / len(batch_metrics)
        train_success = sum(item["success_rate"] for item in batch_metrics) / len(batch_metrics)
        log(f"[train_linear_rl] evaluating epoch {epoch}")
        eval_metrics = evaluate_policy(
            env,
            policy,
            eval_examples,
            max_steps=args.max_steps,
            stop_on_cycle=args.stop_on_cycle,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_avg_reward": train_reward,
            "train_success_rate": train_success,
            "eval_success_rate": eval_metrics["success_rate"],
            "eval_avg_steps": eval_metrics["avg_steps"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        latest_weights_path = args.save_dir / "latest_weights.json"
        log(f"[train_linear_rl] writing latest weights to {latest_weights_path}")
        save_json(latest_weights_path, policy.export_weights())
        save_json(args.save_dir / "history.json", history)

        if eval_metrics["success_rate"] > best_success_rate:
            best_success_rate = eval_metrics["success_rate"]
            log(
                "[train_linear_rl] "
                f"new best eval_success_rate={best_success_rate:.4f}, saving to {best_weights_path}"
            )
            save_json(best_weights_path, policy.export_weights())

    summary = {
        "epochs": args.epochs,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "best_eval_success_rate": best_success_rate,
        "device": device,
        "latest_weights_path": str(args.save_dir / "latest_weights.json"),
        "best_weights_path": str(best_weights_path),
    }
    log("[train_linear_rl] training completed")
    save_json(args.save_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
