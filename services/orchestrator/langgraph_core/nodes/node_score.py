from __future__ import annotations

from typing import Any, Dict, List

from ..llm_tools import LLM
from ..state_types import SessionState
from .base_node import Node


class ScoreNode(Node):
    """并行评分节点（当前串行实现）"""

    def __init__(self, name: str):
        super().__init__(name)

    def run(self, state: SessionState, **_: Any) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        for item_id, ctx in state.item_contexts.items():
            payload = {
                "item_name": ctx.item_name,
                "facts": ctx.facts,
                "themes": ctx.themes,
                "summary": ctx.summary,
                "dialogue": ctx.dialogue,
                "risks": ctx.risks,
                "patient_snapshot": ctx.patient_snapshot,
            }
            result = LLM.call("score_item", payload) or {}
            score = result.get("score", 0)
            items.append({"item_id": item_id, "score": score, "raw": result})
        state.analysis = {"total_score": {"sum": sum(item["score"] for item in items), "items": items}}
        return {"analysis": state.analysis}
