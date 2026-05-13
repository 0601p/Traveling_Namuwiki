from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


ACTIONS_DATASET = "0601p/Traveling_Namuwiki_Actions"
PATHS_DATASET = "0601p/Traveling_Namuwiki_Paths"
RAW_DATASET = "heegyu/namuwiki"

Title = str
Action = str


@dataclass(frozen=True)
class Page:
    title: Title
    actions: Sequence[Action]
    raw: str = ""
