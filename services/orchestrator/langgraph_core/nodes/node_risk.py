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
        user_text = str(kwargs.get("user_text") or "")
        if not user_text:
            return {}
        result = LLM.call("risk_detect", {"text": user_text}) or {}
        risk_level = result.get("risk_level")
        if risk_level == "high":
            return {"risk_level": "high", "message": "检测到高危表述"}
        return {}
