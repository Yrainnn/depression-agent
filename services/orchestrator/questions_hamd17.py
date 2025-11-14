from __future__ import annotations

from typing import Dict, List

from services.orchestrator.config.item_registry import (
    DEFAULT_MAX_SCORES,
    get_item_name,
)


def _build_question_bank() -> Dict[int, Dict[str, List[str]]]:
    bank: Dict[int, Dict[str, List[str]]] = {}
    for item_id in sorted(DEFAULT_MAX_SCORES):
        bank[item_id] = {
            "primary": [get_item_name(item_id)],
            "clarify": {},
        }
    return bank


HAMD17_QUESTION_BANK: Dict[int, Dict[str, List[str]]] = _build_question_bank()
MAX_SCORE: Dict[int, int] = dict(DEFAULT_MAX_SCORES)


def get_first_item() -> int:
    return min(MAX_SCORE)


def get_next_item(current: int) -> int:
    ordered = sorted(MAX_SCORE)
    for item_id in ordered:
        if item_id > current:
            return item_id
    return -1


def pick_primary(item_id: int) -> str:
    return get_item_name(item_id)


def pick_clarify(item_id: int, gap: str) -> str:
    clarifies = HAMD17_QUESTION_BANK.get(item_id, {}).get("clarify", {})
    if isinstance(clarifies, dict):
        for prompts in clarifies.values():
            if isinstance(prompts, list) and prompts:
                return prompts[0]
    return ""


__all__ = [
    "HAMD17_QUESTION_BANK",
    "MAX_SCORE",
    "get_first_item",
    "get_next_item",
    "pick_primary",
    "pick_clarify",
]

