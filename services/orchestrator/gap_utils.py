"""Utilities for detecting missing information in HAMD-17 dialogues."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

# Mapping between internal gap keys and the labels used in clarify prompts.
GAP_LABELS: Dict[str, str] = {
    "frequency": "频次",
    "duration": "持续时间",
    "severity": "严重程度",
    "negation": "是否否定",
    "plan": "是否有计划",
    "safety": "安全保障",
}

FREQUENCY_KEYWORDS: Sequence[str] = (
    "次",
    "天",
    "每天",
    "每日",
    "一周",
    "周内",
    "每周",
    "多数天",
    "大部分时间",
    "几乎每天",
    "经常",
    "常常",
)

FREQUENCY_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"[0-9一二三四五六七八九十半]{1,3}\s*次"),
    re.compile(r"[0-9一二三四五六七八九十半]{1,3}\s*天"),
    re.compile(r"每[天周月]"),
)

DURATION_KEYWORDS: Sequence[str] = (
    "整天",
    "全天",
    "一整天",
    "小时",
    "多长",
    "多久",
    "分钟",
    "半天",
    "一段时间",
    "一阵",
)

DURATION_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"[0-9一二三四五六七八九十半]{1,3}\s*小时"),
    re.compile(r"[0-9一二三四五六七八九十半]{1,3}\s*分钟"),
)

SEVERITY_KEYWORDS: Sequence[str] = (
    "严重",
    "很难",
    "影响",
    "困扰",
    "麻烦",
    "明显",
    "厉害",
    "痛苦",
    "难受",
    "难以",
    "困难",
    "糟糕",
    "糟透",
    "受不了",
    "没办法",
    "无法",
    "不行",
    "下降",
    "降低",
    "加重",
    "更差",
    "越来越",
    "特别",
    "非常",
)

SEVERITY_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"(?:影响|影响到).{0,6}(工作|生活|学习)"),
    re.compile(r"(几乎|总是|一直).{0,4}(这样|如此)"),
)


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def detect_information_gaps(text: Optional[str], item_id: Optional[int] = None) -> List[str]:
    """Return the ordered list of unmet information gaps for a response."""

    lowered = (text or "").lower()
    gaps: List[str] = []

    if not _contains_any(lowered, FREQUENCY_KEYWORDS) and not _matches_any(
        lowered, FREQUENCY_PATTERNS
    ):
        gaps.append("frequency")

    if not _contains_any(lowered, DURATION_KEYWORDS) and not _matches_any(
        lowered, DURATION_PATTERNS
    ):
        gaps.append("duration")

    if not _contains_any(lowered, SEVERITY_KEYWORDS) and not _matches_any(
        lowered, SEVERITY_PATTERNS
    ):
        gaps.append("severity")

    if _has_explicit_negation(lowered):
        gaps.append("negation")

    if item_id == 3:
        if not _contains_any(lowered, ("安全", "陪伴", "有人", "同伴")):
            gaps.insert(0, "safety")
        if not _contains_any(lowered, ("计划", "准备", "安排", "打算")):
            gaps.insert(0, "plan")

    return gaps


def _has_explicit_negation(text: str) -> bool:
    """Return True when the response explicitly denies the symptom."""

    if not text:
        return False

    patterns = [
        r"没有",
        r"並不",
        r"并不",
        r"不再",
        r"不太",
        r"不怎",
        r"不需要",
        r"不想",
        r"不必",
        r"不会",
        r"无需",
        r"不用",
        r"未(?:曾|有|发生|出现)",
        r"从未",
        r"否认",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


__all__ = ["GAP_LABELS", "detect_information_gaps"]
