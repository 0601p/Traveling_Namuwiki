from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from embed import CachedEmbedder, EmbeddingModel
from environment import NamuwikiEnvironment
from similarity import get_vector_metric
from utils import Action, Config, Page

from .base import Model


@dataclass(frozen=True)
class BeamState:
    title: str
    first_action: str
    score_sum: float
    depth: int
    seen: frozenset[str]
    reached_target: bool = False

    def rank_score(self, length_penalty: float, target_bonus: float) -> float:
        normalized = self.score_sum / max(self.depth ** length_penalty, 1.0)
        if self.reached_target:
            return normalized + target_bonus
        return normalized


class BeamSemanticWalk(Model):
    """Semantic-walk baseline with fixed-depth beam-search lookahead."""

    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        *,
        model_config: str | Path | None = None,
        embedding_config: str | Path | None = None,
    ) -> None:
        config = Config(model_config)
        self.avoid_visited = bool(config.value("avoid_visited"))
        self.metric = get_vector_metric(str(config.value("metric")))
        self.actions_path = str(config.value("actions_path"))
        self.actions_split = str(config.value("actions_split"))
        self.beam_width = int(config.value("beam_width"))
        self.search_depth = int(config.value("search_depth"))
        self.length_penalty = float(config.value("length_penalty"))
        self.target_bonus = float(config.value("target_bonus"))
        embed_config = Config(embedding_config)
        self.embedder = EmbeddingModel(config=embed_config.data)
        self.query_prefix = None if isinstance(self.embedder.backend, CachedEmbedder) else self.QUERY_PREFIX
        self.passage_prefix = None if isinstance(self.embedder.backend, CachedEmbedder) else self.PASSAGE_PREFIX
        self.env = NamuwikiEnvironment.from_dataset(
            dataset_path=self.actions_path,
            split=self.actions_split,
            load_raw=False,
        )
        self._history: set[str] = set()

    def begin_episode(self, start_title: str, target: str) -> None:
        del target
        self._history = {start_title}

    def score_actions(self, actions: list[str], target: str) -> list[tuple[str, float]]:
        if not actions:
            return []
        target_vector = self.embedder.get_embed(target, prefix=self.query_prefix)
        candidate_vectors = self.embedder.get_embeds(actions, prefix=self.passage_prefix)
        scores = [self.metric(candidate_vector, target_vector) for candidate_vector in candidate_vectors]
        return list(zip(actions, scores))

    def sample(self, page: Page, target: str) -> Action | None:
        actions = list(page.actions)
        if not actions:
            return None

        if target in actions:
            self._history.add(target)
            return target

        candidates = actions
        if self.avoid_visited:
            unvisited = [action for action in actions if action not in self._history]
            candidates = unvisited or actions

        frontier: list[BeamState] = []
        for action, score in self.score_actions(candidates, target):
            frontier.append(
                BeamState(
                    title=action,
                    first_action=action,
                    score_sum=score,
                    depth=1,
                    seen=frozenset({page.title, action}),
                    reached_target=action == target,
                )
            )
        if not frontier:
            return None

        best = max(
            frontier,
            key=lambda state: state.rank_score(self.length_penalty, self.target_bonus),
        )
        if best.reached_target:
            self._history.add(best.first_action)
            return best.first_action

        for _ in range(1, self.search_depth):
            expanded: list[BeamState] = []
            for state in frontier:
                next_actions = list(self.env.actions(state.title))
                if self.avoid_visited:
                    unvisited = [
                        action
                        for action in next_actions
                        if action not in self._history and action not in state.seen
                    ]
                    candidates = unvisited or [
                        action for action in next_actions if action not in state.seen
                    ]
                else:
                    candidates = [
                        action for action in next_actions if action not in state.seen
                    ]
                for action, score in self.score_actions(candidates, target):
                    expanded.append(
                        BeamState(
                            title=action,
                            first_action=state.first_action,
                            score_sum=state.score_sum + score,
                            depth=state.depth + 1,
                            seen=state.seen | {action},
                            reached_target=action == target,
                        )
                    )
            if not expanded:
                break
            expanded.sort(
                key=lambda state: state.rank_score(self.length_penalty, self.target_bonus),
                reverse=True,
            )
            frontier = expanded[: self.beam_width]
            candidate_best = frontier[0]
            if candidate_best.rank_score(self.length_penalty, self.target_bonus) > best.rank_score(
                self.length_penalty, self.target_bonus
            ):
                best = candidate_best
            if best.reached_target:
                break

        self._history.add(best.first_action)
        return best.first_action
