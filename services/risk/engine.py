from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class RiskAssessment:
    level: str
    triggers: List[str]


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
        hits = [kw for kw in self._high_risk_keywords if kw in text]
        if hits:
            return RiskAssessment(level="high", triggers=hits)
        return RiskAssessment(level="low", triggers=[])


engine = RiskEngine()
