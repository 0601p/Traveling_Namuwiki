from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence


ACTIONS_DATASET = "0601p/Traveling_Namuwiki_Actions"
PATHS_DATASET = "0601p/Traveling_Namuwiki_Paths"
RAW_DATASET = "heegyu/namuwiki"
DEFAULT_LM_BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"

Title = str
Action = str


@dataclass(frozen=True)
class Page:
    title: Title
    actions: Sequence[Action]
    raw: str = ""


TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def normalize_text(text: str) -> str:
    return normalized_text(text)


def tokenize(text: str) -> list[str]:
    return list(token_tuple(text))


def char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = normalize_text(text).replace(" ", "")
    if not compact:
        return set()
    if len(compact) < n:
        return {compact}
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


@lru_cache(maxsize=500_000)
def normalized_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


@lru_cache(maxsize=500_000)
def token_tuple(text: str) -> tuple[str, ...]:
    normalized = normalized_text(text)
    if not normalized:
        return ()
    return tuple(normalized.split())


@lru_cache(maxsize=500_000)
def char_ngram_tuple(text: str, n: int = 3) -> tuple[str, ...]:
    compact = normalized_text(text).replace(" ", "")
    if not compact:
        return ()
    if len(compact) < n:
        return (compact,)
    return tuple({compact[index : index + n] for index in range(len(compact) - n + 1)})


@lru_cache(maxsize=1_000_000)
def cached_similarity(left: str, right: str) -> float:
    left_tokens = token_tuple(left)
    right_tokens = token_tuple(right)
    token_score = overlap_ratio(left_tokens, right_tokens)
    trigram_score = jaccard(char_ngram_tuple(left), char_ngram_tuple(right))
    left_norm = normalized_text(left)
    right_norm = normalized_text(right)
    substring_bonus = 1.0 if left_norm and right_norm and right_norm in left_norm else 0.0
    return 0.45 * token_score + 0.45 * trigram_score + 0.10 * substring_bonus


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def overlap_ratio(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    overlap = sum(min(left_counts[token], right_counts[token]) for token in left_counts)
    return overlap / max(len(left), len(right))
