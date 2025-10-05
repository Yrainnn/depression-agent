"""Utilities for detecting missing information in HAMD-17 dialogues."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

# Mapping between internal gap keys and the labels used in clarify prompts.
GAP_LABELS: Dict[str, str] = {
    "frequency": "频次",
    "duration": "持续时间",
    "severity": "严重程度",
    "negation": "是否否定",
    "plan": "是否有计划",
    "safety": "安全保障",
}


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def detect_information_gaps(text: Optional[str], item_id: Optional[int] = None) -> List[str]:
    """Return the ordered list of unmet information gaps for a response."""

    lowered = (text or "").lower()
    gaps: List[str] = []

    if not _contains_any(
        lowered,
        (
            "次",
            "天",
            "每天",
            "每日",
            "一周",
            "周内",
            "每周",
        ),
    ):
        gaps.append("frequency")

    if not _contains_any(
        lowered,
        (
            "整天",
            "全天",
            "一整天",
            "小时",
            "多长",
            "多久",
            "分钟",
            "半天",
        ),
    ):
        gaps.append("duration")

    if not _contains_any(
        lowered,
        (
            "严重",
            "很难",
            "影响",
            "困扰",
            "麻烦",
        ),
    ):
        gaps.append("severity")

    if _contains_any(lowered, ("没有", "不", "无")):
        gaps.append("negation")

    if item_id == 3:
        if not _contains_any(lowered, ("安全", "陪伴", "有人", "同伴")):
            gaps.insert(0, "safety")
        if not _contains_any(lowered, ("计划", "准备", "安排", "打算")):
            gaps.insert(0, "plan")

    return gaps


__all__ = ["GAP_LABELS", "detect_information_gaps"]
