from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from embed import load_embedding_config


ACTIONS_DATASET = "0601p/Traveling_Namuwiki_Actions"
PATHS_DATASET = "0601p/Traveling_Namuwiki_Paths"
RAW_DATASET = "heegyu/namuwiki"

Title = str
Action = str


class Config:
    """Small wrapper for flat YAML config files."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self.data: dict[str, object] = (
            load_embedding_config(self.path) if self.path is not None else {}
        )

    def value(self, key: str) -> object:
        if key not in self.data:
            source = f" in {self.path}" if self.path is not None else ""
            raise ValueError(f"Missing required config value{source}: {key}")
        return self.data[key]


@dataclass(frozen=True)
class Page:
    title: Title
    actions: Sequence[Action]
    raw: str = ""
