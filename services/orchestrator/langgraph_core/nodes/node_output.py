from __future__ import annotations

from typing import Any, Dict

from ..state_types import SessionState
from ..utils import now_iso
from .base_node import Node


class OutputNode(Node):
    """统一响应输出节点"""

    def __init__(self, name: str):
        super().__init__(name)

    def run(self, state: SessionState, **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = kwargs.get("payload", {}) or {}
        response: Dict[str, Any] = {
            "ts": now_iso(),
            "sid": state.sid,
            "progress": {"index": state.index, "total": state.total},
            "waiting_for_user": state.waiting_for_user,
            "current_item": {"id": state.index, "name": state.current_item_name},
            "current_strategy": state.current_strategy,
            "patient_context": {
                "summary": state.patient_context.conversation_summary,
                "themes": list(state.patient_context.narrative_themes),
                "risks": list(state.patient_context.active_risks),
            },
        }
        response.update(payload)
        if state.analysis and "analysis" not in response:
            response["analysis"] = state.analysis
        return response
