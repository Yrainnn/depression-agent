from __future__ import annotations

from typing import Any, Dict

from ..nodes.node_risk import RiskNode
from ..state_types import SessionState


class RiskSubgraph:
    """风险门控子图：占位实现，兼容 future 扩展"""

    def __init__(self, risk_node: RiskNode) -> None:
        self.risk_node = risk_node

    def __call__(self, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        session: SessionState = graph_state["session"]
        text = graph_state.get("text")
        payload = graph_state.get("payload", {}) or {}
        result = self.risk_node.run(session, user_text=text)
        if result:
            payload.update(result)
        graph_state["payload"] = payload
        graph_state["risk_result"] = result
        return graph_state
