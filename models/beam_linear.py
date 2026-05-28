from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from environment import NamuwikiEnvironment
from utils import Action, Config, Page

from .base import Model
from .linear import LinearModel


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


class BeamLinear(Model):
    """Look ahead multiple steps with beam search using linear local scores."""

    def __init__(
        self,
        *,
        model_config: str | Path | None = None,
        embedding_config: str | Path | None = None,
    ) -> None:
        config = Config(model_config)
        self.base = LinearModel(
            model_config=model_config,
            embedding_config=embedding_config,
        )
        self.actions_path = str(config.value("actions_path"))
        self.actions_split = str(config.value("actions_split"))
        self.beam_width = int(config.value("beam_width"))
        self.search_depth = int(config.value("search_depth"))
        self.length_penalty = float(config.value("length_penalty"))
        self.target_bonus = float(config.value("target_bonus"))
        self.avoid_visited = bool(config.value("avoid_visited"))
        self.env = NamuwikiEnvironment.from_dataset(
            dataset_path=self.actions_path,
            split=self.actions_split,
            load_raw=False,
        )
        self._episode_seen: set[str] = set()

    def begin_episode(self, start_title: str, target: str) -> None:
        del target
        self._episode_seen = {start_title}

    def sample(self, page: Page, target: str) -> Action | None:
        actions = list(page.actions)
        if not actions:
            return None

        if target in actions:
            self._episode_seen.add(target)
            return target

        frontier: list[BeamState] = []
        for action in actions:
            if self.avoid_visited and action in self._episode_seen:
                continue
            score = self.base.score(action, target)
            if score is None:
                continue
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
            self._episode_seen.add(best.first_action)
            return best.first_action

        for _ in range(1, self.search_depth):
            expanded: list[BeamState] = []
            for state in frontier:
                next_actions = list(self.env.actions(state.title))
                for action in next_actions:
                    if self.avoid_visited and action in self._episode_seen:
                        continue
                    if action in state.seen:
                        continue
                    score = self.base.score(action, target)
                    if score is None:
                        continue
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
                key=lambda state: state.rank_score(
                    self.length_penalty,
                    self.target_bonus,
                ),
                reverse=True,
            )
            frontier = expanded[: self.beam_width]
            candidate_best = frontier[0]
            if candidate_best.rank_score(self.length_penalty, self.target_bonus) > best.rank_score(
                self.length_penalty,
                self.target_bonus,
            ):
                best = candidate_best
            if best.reached_target:
                break

        self._episode_seen.add(best.first_action)
        return best.first_action
