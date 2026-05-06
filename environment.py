from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from datasets import load_dataset

from models import Model
from utils import ACTIONS_DATASET, Action, Page, Title


Graph = dict[Title, list[Action]]


@dataclass(frozen=True)
class SearchResult:
    start_title: Title
    target_title: Title
    visited: list[Title]
    stopped_reason: str

    @property
    def end_title(self) -> Title:
        return self.visited[-1]

    @property
    def steps_taken(self) -> int:
        return len(self.visited) - 1

    @property
    def reached_target(self) -> bool:
        return self.stopped_reason == "target_reached"


class NamuwikiEnvironment:
    """State/action environment backed by Traveling_Namuwiki_Actions."""

    def __init__(self, graph: Mapping[Title, Sequence[Action]]) -> None:
        self.graph: Graph = {
            title: list(dict.fromkeys(actions)) for title, actions in graph.items()
        }

    @classmethod
    def from_dataset(
        cls,
        dataset_path: str = ACTIONS_DATASET,
        *,
        split: str = "train",
    ) -> NamuwikiEnvironment:
        graph: Graph = {}
        for row in load_dataset(dataset_path, split=split):
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            graph[title] = list(row["actions"])
        return cls(graph)

    def actions(self, title: Title) -> Sequence[Action]:
        return self.graph.get(title, [])

    def walk(
        self,
        *,
        start_title: Title,
        target_title: Title,
        model: Model,
        max_steps: int,
        stop_on_cycle: bool = True,
    ) -> SearchResult:
        visited = [start_title]
        seen = {start_title}
        current = start_title

        for _ in range(max_steps):
            page = Page(title=current, actions=self.actions(current))
            next_title = model.sample(page, target_title)

            if next_title is None:
                return SearchResult(
                    start_title, target_title, visited, "no_actions"
                )

            visited.append(next_title)
            if next_title == target_title:
                return SearchResult(
                    start_title, target_title, visited, "target_reached"
                )

            if stop_on_cycle and next_title in seen:
                return SearchResult(start_title, target_title, visited, "cycle")

            seen.add(next_title)
            current = next_title

        return SearchResult(start_title, target_title, visited, "max_steps")
