from __future__ import annotations

from typing import Dict, Any

from ..nodes.node_clarify import ClarifyNode
from ..nodes.node_strategy import StrategyNode
from ..nodes.node_update import UpdateNode
from ..state_types import SessionState


class StrategySubgraph:
    """策略执行子图（封装澄清与上下文更新）"""

    def __init__(
        self,
        strategy_node: StrategyNode,
        clarify_node: ClarifyNode,
        update_node: UpdateNode,
    ) -> None:
        self.strategy_node = strategy_node
        self.clarify_node = clarify_node
        self.update_node = update_node

    def __call__(self, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        session: SessionState = graph_state["session"]
        role = graph_state.get("role", "agent")
        text = graph_state.get("text")
        payload = graph_state.get("payload", {}) or {}

        if role == "agent":
            result = self.strategy_node.run(session, role="agent")
            payload.update(result)
        else:
            clarify = self.clarify_node.run(session, user_text=text)
            payload.update({k: v for k, v in clarify.items() if v is not None})
            update = self.update_node.run(session, user_text=text, **clarify)
            payload.update(update)
        graph_state["payload"] = payload
        return graph_state
