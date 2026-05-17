from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from environment import NamuwikiEnvironment, SearchResult
from evaluate_paths import iter_path_examples, limited, mean
from models.neural_target_a2c import NeuralTargetActorCritic
from models.beam_her_reranker import BeamHerReranker, load_beam_reranker_checkpoint
from utils import ACTIONS_DATASET, DEFAULT_LM_BACKBONE, PATHS_DATASET, cached_similarity


@dataclass(frozen=True)
class BeamState:
    current: str
    visited: tuple[str, ...]
    score: float
    stopped_reason: str | None = None

    @property
    def done(self) -> bool:
        return self.stopped_reason is not None


class LmScorer:
    def __init__(self, *, model_name: str, max_length: int, device: torch.device) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def similarity(self, left: str, right: str) -> float:
        embeddings = self.encode([left, right])
        embeddings = F.normalize(embeddings, dim=1)
        return float(torch.matmul(embeddings[0], embeddings[1]).cpu())

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        missing = [text for text in dict.fromkeys(texts) if text not in self.cache]
        if missing:
            batch = self.tokenizer(
                missing,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(outputs.dtype)
            embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            for text, embedding in zip(missing, embeddings.detach().cpu()):
                self.cache[text] = embedding
        return torch.stack([self.cache[text] for text in texts]).to(self.device)


class ValueScorer:
    def __init__(self, *, checkpoint: Path, device: torch.device) -> None:
        payload = torch.load(checkpoint, map_location=device)
        config = payload.get("config", {})
        self.network = NeuralTargetActorCritic(**config).to(device)
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()
        self.device = device

    @torch.no_grad()
    def value(self, current: str, target: str) -> float:
        embeddings = self.network.encode_texts([current, target], device=self.device)
        value = self.network.value(embeddings[0:1], embeddings[1:2]).squeeze()
        return float(value.cpu())

    @torch.no_grad()
    def policy_score(self, current: str, action: str, target: str) -> float:
        embeddings = self.network.encode_texts([current, target, action], device=self.device)
        logits = self.network.actor_logits(
            current_embedding=embeddings[0:1],
            target_embedding=embeddings[1:2],
            action_embeddings=embeddings[2:],
        )
        return float(logits.squeeze().cpu())


class BeamHerRerankerScorer:
    def __init__(self, *, checkpoint: Path, device: torch.device) -> None:
        self.model = load_beam_reranker_checkpoint(path=checkpoint, device=device)
        self.device = device

    @torch.no_grad()
    def score(
        self,
        *,
        env: NamuwikiEnvironment,
        current: str,
        action: str,
        target: str,
        visited: Sequence[str],
        two_hop_weight: float,
    ) -> float:
        features = reranker_scalar_features(
            env=env,
            current=current,
            target=target,
            actions=[action],
            visited=visited,
            two_hop_weight=two_hop_weight,
            device=self.device,
        )
        scores = self.model.scores(
            current=current,
            target=target,
            actions=[action],
            scalar_features=features,
            device=self.device,
        )
        return float(scores.squeeze().cpu())

    @torch.no_grad()
    def scores(
        self,
        *,
        env: NamuwikiEnvironment,
        current: str,
        actions: Sequence[str],
        target: str,
        visited: Sequence[str],
        two_hop_weight: float,
    ) -> list[float]:
        features = reranker_scalar_features(
            env=env,
            current=current,
            target=target,
            actions=actions,
            visited=visited,
            two_hop_weight=two_hop_weight,
            device=self.device,
        )
        scores = self.model.scores(
            current=current,
            target=target,
            actions=actions,
            scalar_features=features,
            device=self.device,
        )
        return [float(score) for score in scores.detach().cpu()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lexical/LM/value beam search baselines.")
    parser.add_argument(
        "--method",
        choices=[
            "lexical",
            "twohop",
            "hybrid",
            "value",
            "residual_her",
            "twohop_residual_her",
            "twohop_her_reranker",
        ],
        required=True,
    )
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    parser.add_argument("--actions-path", default=ACTIONS_DATASET)
    parser.add_argument("--paths-path", default=PATHS_DATASET)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--expand-top-k", type=int, default=16)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--two-hop-weight", type=float, default=0.4)
    parser.add_argument("--lm-weight", type=float, default=0.5)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--policy-weight", type=float, default=0.25)
    parser.add_argument("--cycle-penalty", type=float, default=0.5)
    parser.add_argument("--model-name", default=DEFAULT_LM_BACKBONE)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument(
        "--value-checkpoint",
        type=Path,
        default=Path("outputs/runs/residual_hindsight_target_a2c_v3_5000_alpha5/residual_hindsight_target_a2c_v3.pt"),
    )
    parser.add_argument(
        "--reranker-checkpoint",
        type=Path,
        default=Path("outputs/runs/twohop_her_beam_reranker/twohop_her_beam_reranker.pt"),
    )
    parser.add_argument("--reranker-weight", type=float, default=0.25)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()
    if args.predictions_output is None:
        args.predictions_output = Path("outputs") / (
            f"{args.split}_beam_{args.method}_predictions.jsonl"
        )
    if args.metrics_output is None:
        args.metrics_output = Path("outputs") / (
            f"{args.split}_beam_{args.method}_metrics.json"
        )
    return args


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"device": str(device), "method": args.method}), flush=True)

    env = NamuwikiEnvironment.from_dataset(args.actions_path)
    lm_scorer = None
    value_scorer = None
    reranker_scorer = None
    if args.method == "hybrid":
        lm_scorer = LmScorer(model_name=args.model_name, max_length=args.max_length, device=device)
    if args.method in {"value", "residual_her", "twohop_residual_her"}:
        value_scorer = ValueScorer(checkpoint=args.value_checkpoint, device=device)
    if args.method == "twohop_her_reranker":
        reranker_scorer = BeamHerRerankerScorer(
            checkpoint=args.reranker_checkpoint,
            device=device,
        )

    metrics = evaluate(
        args=args,
        env=env,
        lm_scorer=lm_scorer,
        value_scorer=value_scorer,
        reranker_scorer=reranker_scorer,
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def evaluate(
    *,
    args: argparse.Namespace,
    env: NamuwikiEnvironment,
    lm_scorer: LmScorer | None,
    value_scorer: ValueScorer | None,
    reranker_scorer: BeamHerRerankerScorer | BeamHerReranker | None = None,
) -> dict:
    failure_distance = args.max_steps + 1
    distances_by_gold: dict[int, list[float]] = defaultdict(list)
    counts_by_gold: Counter[int] = Counter()
    success_by_gold: Counter[int] = Counter()
    stopped_reasons: Counter[str] = Counter()
    total = 0

    predictions_file = args.predictions_output.open("w", encoding="utf-8")
    try:
        for example in limited(iter_path_examples(args.paths_path, args.split), args.limit):
            total += 1
            counts_by_gold[example.min_distance] += 1
            result = beam_search(
                env=env,
                start=example.start_title,
                target=example.target_title,
                method=args.method,
                beam_size=args.beam_size,
                expand_top_k=args.expand_top_k,
                max_steps=args.max_steps,
                lexical_weight=args.lexical_weight,
                two_hop_weight=args.two_hop_weight,
                lm_weight=args.lm_weight,
                value_weight=args.value_weight,
                policy_weight=args.policy_weight,
                cycle_penalty=args.cycle_penalty,
                lm_scorer=lm_scorer,
                value_scorer=value_scorer,
                reranker_scorer=reranker_scorer,
                reranker_weight=args.reranker_weight,
            )
            stopped_reasons[result.stopped_reason] += 1
            if result.reached_target:
                model_distance = result.steps_taken
                success_by_gold[example.min_distance] += 1
            else:
                model_distance = failure_distance
            distances_by_gold[example.min_distance].append(float(model_distance))
            predictions_file.write(
                json.dumps(
                    {
                        "start_title": example.start_title,
                        "target_title": example.target_title,
                        "gold_min_distance": example.min_distance,
                        "model_distance": model_distance,
                        "reached": result.reached_target,
                        "stopped_reason": result.stopped_reason,
                        "visited": result.visited,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        predictions_file.close()

    return {
        "split": args.split,
        "model": f"beam_{args.method}",
        "total": total,
        "beam_size": args.beam_size,
        "expand_top_k": args.expand_top_k,
        "score_by_min_distance": {
            distance: mean(distances_by_gold[distance])
            for distance in sorted(counts_by_gold)
        },
        "success_rate_by_min_distance": {
            distance: success_by_gold[distance] / counts_by_gold[distance]
            for distance in sorted(counts_by_gold)
        },
        "count_by_min_distance": dict(counts_by_gold),
        "failure_distance": failure_distance,
        "stopped_reasons": dict(stopped_reasons),
    }


def beam_search(
    *,
    env: NamuwikiEnvironment,
    start: str,
    target: str,
    method: str,
    beam_size: int,
    expand_top_k: int,
    max_steps: int,
    lexical_weight: float,
    two_hop_weight: float,
    lm_weight: float,
    value_weight: float,
    policy_weight: float,
    cycle_penalty: float,
    lm_scorer: LmScorer | None,
    value_scorer: ValueScorer | None,
    reranker_scorer: BeamHerRerankerScorer | BeamHerReranker | None = None,
    reranker_weight: float = 0.25,
) -> SearchResult:
    beam = [BeamState(current=start, visited=(start,), score=0.0)]
    completed: list[BeamState] = []

    for _ in range(max_steps):
        candidates: list[BeamState] = []
        for state in beam:
            if state.done:
                completed.append(state)
                continue
            actions = list(env.actions(state.current))
            if not actions:
                completed.append(
                    BeamState(state.current, state.visited, state.score, "no_actions")
                )
                continue
            ranked_actions = sorted(
                actions,
                key=lambda action: cached_similarity(action, target),
                reverse=True,
            )[:expand_top_k]
            reranker_scores: dict[str, float] = {}
            if method == "twohop_her_reranker":
                if reranker_scorer is None:
                    raise ValueError("twohop_her_reranker method requires reranker_scorer")
                reranker_values = score_actions_with_reranker(
                    reranker_scorer=reranker_scorer,
                    env=env,
                    current=state.current,
                    actions=ranked_actions,
                    target=target,
                    visited=state.visited,
                    two_hop_weight=two_hop_weight,
                )
                reranker_scores = dict(zip(ranked_actions, reranker_values))
            for action in ranked_actions:
                if method == "twohop_her_reranker":
                    next_score = state.score + twohop_reranker_action_score(
                        env=env,
                        current=state.current,
                        action=action,
                        target=target,
                        lexical_weight=lexical_weight,
                        two_hop_weight=two_hop_weight,
                        reranker_weight=reranker_weight,
                        reranker_score=reranker_scores[action],
                    )
                else:
                    next_score = state.score + action_score(
                        method=method,
                        env=env,
                        current=state.current,
                        action=action,
                        target=target,
                        lexical_weight=lexical_weight,
                        two_hop_weight=two_hop_weight,
                        lm_weight=lm_weight,
                        value_weight=value_weight,
                        policy_weight=policy_weight,
                        visited=state.visited,
                        lm_scorer=lm_scorer,
                        value_scorer=value_scorer,
                        reranker_scorer=reranker_scorer,
                        reranker_weight=reranker_weight,
                    )
                if action in state.visited:
                    next_score -= cycle_penalty
                    candidates.append(
                        BeamState(action, (*state.visited, action), next_score, "cycle")
                    )
                    continue
                if action == target:
                    candidates.append(
                        BeamState(
                            action,
                            (*state.visited, action),
                            next_score + 10.0,
                            "target_reached",
                        )
                    )
                    continue
                candidates.append(
                    BeamState(action, (*state.visited, action), next_score)
                )
        if not candidates:
            break
        candidates.sort(key=lambda state: state.score, reverse=True)
        completed.extend([state for state in candidates if state.done])
        successful = [state for state in completed if state.stopped_reason == "target_reached"]
        if successful:
            best = max(successful, key=lambda state: state.score)
            return SearchResult(start, target, list(best.visited), "target_reached")
        beam = [state for state in candidates if not state.done][:beam_size]
        if not beam:
            break

    pool = completed + beam
    if not pool:
        return SearchResult(start, target, [start], "no_actions")
    best = max(pool, key=lambda state: state.score)
    reason = best.stopped_reason or "max_steps"
    return SearchResult(start, target, list(best.visited), reason)


def action_score(
    *,
    method: str,
    env: NamuwikiEnvironment,
    current: str,
    action: str,
    target: str,
    lexical_weight: float,
    two_hop_weight: float,
    lm_weight: float,
    value_weight: float,
    policy_weight: float,
    visited: Sequence[str],
    lm_scorer: LmScorer | None,
    value_scorer: ValueScorer | None,
    reranker_scorer: BeamHerRerankerScorer | BeamHerReranker | None,
    reranker_weight: float,
) -> float:
    lexical = cached_similarity(action, target)
    if method == "lexical":
        return lexical_weight * lexical
    if method == "twohop":
        neighbors = list(env.actions(action))
        two_hop = 0.0
        if neighbors:
            two_hop = max(cached_similarity(neighbor, target) for neighbor in neighbors[:64])
        return lexical_weight * lexical + two_hop_weight * two_hop
    if method == "hybrid":
        if lm_scorer is None:
            raise ValueError("hybrid method requires lm_scorer")
        return lexical_weight * lexical + lm_weight * lm_scorer.similarity(action, target)
    if method == "value":
        if value_scorer is None:
            raise ValueError("value method requires value_scorer")
        progress = cached_similarity(action, target) - cached_similarity(current, target)
        return lexical_weight * lexical + 0.5 * progress + value_weight * value_scorer.value(action, target)
    if method == "residual_her":
        if value_scorer is None:
            raise ValueError("residual_her method requires value_scorer")
        progress = cached_similarity(action, target) - cached_similarity(current, target)
        return (
            lexical_weight * lexical
            + 0.5 * progress
            + policy_weight * value_scorer.policy_score(current, action, target)
            + value_weight * value_scorer.value(action, target)
        )
    if method == "twohop_residual_her":
        if value_scorer is None:
            raise ValueError("twohop_residual_her method requires value_scorer")
        neighbors = list(env.actions(action))
        two_hop = 0.0
        if neighbors:
            two_hop = max(cached_similarity(neighbor, target) for neighbor in neighbors[:64])
        progress = cached_similarity(action, target) - cached_similarity(current, target)
        return (
            lexical_weight * lexical
            + two_hop_weight * two_hop
            + 0.5 * progress
            + policy_weight * value_scorer.policy_score(current, action, target)
            + value_weight * value_scorer.value(action, target)
        )
    if method == "twohop_her_reranker":
        if reranker_scorer is None:
            raise ValueError("twohop_her_reranker method requires reranker_scorer")
        neighbors = list(env.actions(action))
        two_hop = 0.0
        if neighbors:
            two_hop = max(cached_similarity(neighbor, target) for neighbor in neighbors[:64])
        reranker_score = score_with_reranker(
            reranker_scorer=reranker_scorer,
            env=env,
            current=current,
            action=action,
            target=target,
            visited=visited,
            two_hop_weight=two_hop_weight,
        )
        return lexical_weight * lexical + two_hop_weight * two_hop + reranker_weight * reranker_score
    raise ValueError(f"Unknown method: {method}")


def twohop_reranker_action_score(
    *,
    env: NamuwikiEnvironment,
    current: str,
    action: str,
    target: str,
    lexical_weight: float,
    two_hop_weight: float,
    reranker_weight: float,
    reranker_score: float,
) -> float:
    del current
    lexical = cached_similarity(action, target)
    neighbors = list(env.actions(action))
    two_hop = 0.0
    if neighbors:
        two_hop = max(cached_similarity(neighbor, target) for neighbor in neighbors[:64])
    return lexical_weight * lexical + two_hop_weight * two_hop + reranker_weight * reranker_score


def reranker_scalar_features(
    *,
    env: NamuwikiEnvironment,
    current: str,
    target: str,
    actions: Sequence[str],
    visited: Sequence[str],
    two_hop_weight: float,
    device: torch.device,
) -> torch.Tensor:
    import math

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


@torch.no_grad()
def score_with_reranker(
    *,
    reranker_scorer: BeamHerRerankerScorer | BeamHerReranker,
    env: NamuwikiEnvironment,
    current: str,
    action: str,
    target: str,
    visited: Sequence[str],
    two_hop_weight: float,
) -> float:
    if isinstance(reranker_scorer, BeamHerRerankerScorer):
        return reranker_scorer.score(
            env=env,
            current=current,
            action=action,
            target=target,
            visited=visited,
            two_hop_weight=two_hop_weight,
        )
    device = next(reranker_scorer.parameters()).device
    features = reranker_scalar_features(
        env=env,
        current=current,
        target=target,
        actions=[action],
        visited=visited,
        two_hop_weight=two_hop_weight,
        device=device,
    )
    scores = reranker_scorer.scores(
        current=current,
        target=target,
        actions=[action],
        scalar_features=features,
        device=device,
    )
    return float(scores.squeeze().cpu())


@torch.no_grad()
def score_actions_with_reranker(
    *,
    reranker_scorer: BeamHerRerankerScorer | BeamHerReranker,
    env: NamuwikiEnvironment,
    current: str,
    actions: Sequence[str],
    target: str,
    visited: Sequence[str],
    two_hop_weight: float,
) -> list[float]:
    if isinstance(reranker_scorer, BeamHerRerankerScorer):
        return reranker_scorer.scores(
            env=env,
            current=current,
            actions=actions,
            target=target,
            visited=visited,
            two_hop_weight=two_hop_weight,
        )
    device = next(reranker_scorer.parameters()).device
    features = reranker_scalar_features(
        env=env,
        current=current,
        target=target,
        actions=actions,
        visited=visited,
        two_hop_weight=two_hop_weight,
        device=device,
    )
    scores = reranker_scorer.scores(
        current=current,
        target=target,
        actions=actions,
        scalar_features=features,
        device=device,
    )
    return [float(score) for score in scores.detach().cpu()]


if __name__ == "__main__":
    main()
