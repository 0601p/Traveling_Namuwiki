from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import deque
from collections import OrderedDict
from typing import Mapping

from embed import EmbeddingModel
from environment import NamuwikiEnvironment
from similarity import cached_similarity, cosine
from utils import Config


REWARD_SHAPING_MODES = ("none", "lexical", "cosine", "hybrid", "distance", "distance_hybrid")


@dataclass
class RewardShaper:
    mode: str = "none"
    coef: float = 0.0
    embedder: EmbeddingModel | None = None
    distance_helper: DistanceToTarget | None = None

    def enabled(self) -> bool:
        return self.mode != "none" and self.coef != 0.0

    def delta(self, *, current: str, next_title: str, target: str) -> float:
        if not self.enabled():
            return 0.0
        current_score = self._score(current, target)
        next_score = self._score(next_title, target)
        return self.coef * (next_score - current_score)

    def _score(self, source: str, target: str) -> float:
        if self.mode == "lexical":
            return cached_similarity(source, target)
        if self.mode == "cosine":
            return self._cosine_score(source, target)
        if self.mode == "hybrid":
            lexical = cached_similarity(source, target)
            semantic = self._cosine_score(source, target)
            return 0.5 * (lexical + semantic)
        if self.mode == "distance":
            return self._distance_score(source, target)
        if self.mode == "distance_hybrid":
            lexical = cached_similarity(source, target)
            distance = self._distance_score(source, target)
            return 0.5 * (lexical + distance)
        return 0.0

    def _cosine_score(self, source: str, target: str) -> float:
        if self.embedder is None:
            raise ValueError("Cosine reward shaping requires an embedding config")
        source_embedding = self.embedder.get_embed(source)
        target_embedding = self.embedder.get_embed(target)
        return cosine(source_embedding, target_embedding)

    def _distance_score(self, source: str, target: str) -> float:
        if self.distance_helper is None:
            raise ValueError("Distance reward shaping requires environment access")
        return self.distance_helper.score(source, target)


class DistanceToTarget:
    def __init__(
        self,
        graph: Mapping[str, list[str]],
        *,
        max_cached_targets: int = 1,
    ) -> None:
        self.reverse_graph: dict[str, list[str]] = {}
        for title, actions in graph.items():
            self.reverse_graph.setdefault(title, [])
            for action in actions:
                self.reverse_graph.setdefault(action, []).append(title)
        self.max_cached_targets = max(1, int(max_cached_targets))
        self._distance_cache: OrderedDict[str, dict[str, int]] = OrderedDict()

    def score(self, source: str, target: str) -> float:
        distance = self.distance(source, target)
        if distance is None:
            return 0.0
        return 1.0 / (1.0 + distance)

    def distance(self, source: str, target: str) -> int | None:
        if target not in self._distance_cache:
            self._distance_cache[target] = self._bfs(target)
            self._distance_cache.move_to_end(target)
            while len(self._distance_cache) > self.max_cached_targets:
                self._distance_cache.popitem(last=False)
        else:
            self._distance_cache.move_to_end(target)
        return self._distance_cache[target].get(source)

    def _bfs(self, target: str) -> dict[str, int]:
        distances = {target: 0}
        queue: deque[str] = deque([target])
        while queue:
            current = queue.popleft()
            for predecessor in self.reverse_graph.get(current, []):
                if predecessor in distances:
                    continue
                distances[predecessor] = distances[current] + 1
                queue.append(predecessor)
        return distances


def build_reward_shaper(
    *,
    mode: str,
    coef: float,
    env: NamuwikiEnvironment | None = None,
    embedding_config: str | Path | None = None,
    distance_cache_size: int = 1,
) -> RewardShaper:
    normalized = mode.strip().lower()
    if normalized not in REWARD_SHAPING_MODES:
        available = ", ".join(REWARD_SHAPING_MODES)
        raise ValueError(f"Unknown reward shaping mode: {mode!r}. Available: {available}")
    if normalized == "none" or coef == 0.0:
        return RewardShaper(mode="none", coef=0.0)

    embedder = None
    if normalized in {"cosine", "hybrid"}:
        if embedding_config is None:
            raise ValueError(
                f"Reward shaping mode {normalized!r} requires --embedding-config"
            )
        embedder = EmbeddingModel(config=Config(embedding_config).data)
    distance_helper = None
    if normalized in {"distance", "distance_hybrid"}:
        if env is None:
            raise ValueError(
                f"Reward shaping mode {normalized!r} requires environment access"
            )
        distance_helper = DistanceToTarget(
            env.graph,
            max_cached_targets=distance_cache_size,
        )
    return RewardShaper(
        mode=normalized,
        coef=coef,
        embedder=embedder,
        distance_helper=distance_helper,
    )
