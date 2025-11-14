from __future__ import annotations

from typing import Any, Dict

from ..llm_tools import LLM
from ..state_types import SessionState
from .base_node import Node


class RiskNode(Node):
    """风险门控节点：预留风险处理子图"""

    def __init__(self, name: str):
        super().__init__(name)

    def run(self, state: SessionState, **kwargs: Any) -> Dict[str, Any]:
        user_text = str(kwargs.get("user_text") or "").strip()
        if not user_text:
            return {}

        raw_result = LLM.call("risk_detect", {"text": user_text}) or {}
        if not isinstance(raw_result, dict):
            raw_result = {}

        risk_level = str(raw_result.get("risk_level") or "none").lower()
        if risk_level not in {"none", "low", "medium", "high"}:
            risk_level = "none"

        normalized: Dict[str, Any] = {"risk_level": risk_level}

        triggers = raw_result.get("triggers") or raw_result.get("risk_triggers")
        if isinstance(triggers, (list, tuple)):
            cleaned = [str(trigger).strip() for trigger in triggers if str(trigger).strip()]
            if cleaned:
                normalized["risk_triggers"] = cleaned

        reason = raw_result.get("reason") or raw_result.get("rationale")
        if isinstance(reason, str) and reason.strip():
            normalized["risk_reason"] = reason.strip()

        advice = raw_result.get("advice")
        if isinstance(advice, str) and advice.strip():
            normalized["risk_advice"] = advice.strip()

        if risk_level == "high":
            normalized.setdefault("message", "检测到高危表述，请立即转接人工支持。")

        return normalized
