from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import re


NEGATION_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"没有",
        r"並?不",
        r"并?不",
        r"不再",
        r"不想",
        r"不会",
        r"不會",
        r"从未",
        r"沒打算",
        r"没打算",
        r"否认",
        r"無",
    ]
]

NEGATION_WINDOW = 6


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def is_negated(text: str, trigger: str) -> bool:
    """Return True when the trigger appears near a negation cue."""

    normalized = _normalize(text)
    if not trigger:
        return False
    try:
        pattern = re.compile(re.escape(trigger))
    except re.error:
        return False

    has_match = False
    for match in pattern.finditer(normalized):
        has_match = True
        start, end = match.span()
        left = normalized[max(0, start - NEGATION_WINDOW) : start]
        right = normalized[end : min(len(normalized), end + NEGATION_WINDOW)]
        window = f"{left}{right}"
        if not any(pat.search(window) for pat in NEGATION_PATTERNS):
            return False
    return has_match


@dataclass
class RiskAssessment:
    level: str
    triggers: List[str]
    reason: Optional[str] = None


class RiskEngine:
    """Keyword-based risk detection engine."""

    def __init__(self) -> None:
        self._high_risk_keywords = [
            "自杀",
            "结束生命",
            "无法活下去",
            "计划",
            "工具",
        ]

    def evaluate(self, text: str) -> RiskAssessment:
        normalized = _normalize(text)
        if not normalized:
            return RiskAssessment(level="low", triggers=[], reason=None)

        hits: List[str] = []
        for keyword in self._high_risk_keywords:
            if keyword not in normalized:
                continue
            if is_negated(text, keyword):
                continue
            hits.append(keyword)

        if hits:
            return RiskAssessment(
                level="high",
                triggers=hits,
                reason="positive_trigger_no_negation",
            )
        return RiskAssessment(level="low", triggers=[], reason="no_trigger")


engine = RiskEngine()
