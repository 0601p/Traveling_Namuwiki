from __future__ import annotations

import hashlib
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from dataset_utils import load_rows
from models.base import Model
from utils import ACTIONS_DATASET, RAW_DATASET, Action, Page, Title


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

    def __init__(
        self,
        graph: Mapping[Title, Sequence[Action]],
        raws: Mapping[Title, str] | None = None,
    ) -> None:
        self.graph: Graph = {
            title: list(dict.fromkeys(actions)) for title, actions in graph.items()
        }
        self.raws: dict[Title, str] = dict(raws or {})

    @classmethod
    def from_dataset(
        cls,
        dataset_path: str = ACTIONS_DATASET,
        *,
        split: str = "train",
        raw_dataset_path: str = RAW_DATASET,
        load_raw: bool = False,
    ) -> NamuwikiEnvironment:
        cache_path = cls._cache_path(
            dataset_path=dataset_path,
            split=split,
            load_raw=load_raw,
        )
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            return cls(payload["graph"], payload.get("raws"))

        graph: Graph = {}
        for row in load_rows(dataset_path, split=split):
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            graph[title] = list(row["actions"])

        raws: dict[Title, str] = {}
        if load_raw:
            for row in load_rows(raw_dataset_path, split=split):
                title = str(row.get("title") or "").strip()
                if title in graph and title not in raws:
                    raws[title] = row["text"]

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump({"graph": graph, "raws": raws}, handle)

        return cls(graph, raws)

    @staticmethod
    def _cache_path(
        *,
        dataset_path: str,
        split: str,
        load_raw: bool,
    ) -> Path:
        slug = re.sub(r"[^0-9A-Za-z._-]+", "_", dataset_path).strip("._")
        digest = hashlib.sha1(dataset_path.encode("utf-8")).hexdigest()[:10]
        raw_suffix = "_raw" if load_raw else ""
        filename = f"{slug or 'dataset'}_{digest}_{split}{raw_suffix}.pkl"
        return Path(".cache") / filename

    def actions(self, title: Title) -> Sequence[Action]:
        return self.graph.get(title, [])

    def raw(self, title: Title) -> str:
        return self.raws.get(title, "")

    def walk(
        self,
        *,
        start_title: Title,
        target_title: Title,
        model: Model,
        max_steps: int,
        stop_on_cycle: bool = True,
    ) -> SearchResult:
        model.reset(start_title=start_title, target_title=target_title)
        visited = [start_title]
        seen = {start_title}
        current = start_title

        for _ in range(max_steps):
            page = Page(
                title=current,
                actions=self.actions(current),
                raw=self.raw(current),
            )
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
