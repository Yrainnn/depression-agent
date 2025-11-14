from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph  # type: ignore

from ..nodes.node_init import InitNode
from ..nodes.node_output import OutputNode
from ..nodes.node_strategy import StrategyNode
from ..nodes.node_clarify import ClarifyNode
from ..nodes.node_risk import RiskNode
from ..nodes.node_update import UpdateNode
from ..state_types import SessionState
from .subgraph_risk import RiskSubgraph
from .subgraph_strategy import StrategySubgraph


def _wrap(node, **kwargs):
    def _inner(graph_state: Dict[str, Any]) -> Dict[str, Any]:
        session: SessionState = graph_state["session"]
        extra = dict(kwargs)
        extra.update({"role": graph_state.get("role"), "user_text": graph_state.get("text")})
        result = node.run(session, **extra)
        payload = graph_state.get("payload", {}) or {}
        if isinstance(result, dict):
            payload.update(result)
        graph_state["payload"] = payload
        return graph_state

    return _inner


def create_state_graph(template_root: str) -> StateGraph:
    """构建 LangGraph 主图的声明式结构（未编译）。"""

    init_node = InitNode("init_context", template_root)
    strategy_node = StrategyNode("strategy_router")
    clarify_node = ClarifyNode("clarify")
    update_node = UpdateNode("update_context")
    risk_node = RiskNode("risk_gate")
    output_node = OutputNode("output")

    strategy_subgraph = StrategySubgraph(strategy_node, clarify_node, update_node)
    risk_subgraph = RiskSubgraph(risk_node)

    graph = StateGraph(dict)
    graph.add_node("init_context", _wrap(init_node))
    graph.add_node("risk_gate", risk_subgraph)
    graph.add_node("strategy_flow", strategy_subgraph)
    graph.add_node("agent_strategy", strategy_subgraph)
    graph.add_node("output", lambda state: _wrap(output_node, payload=state.get("payload", {}))(state))

    graph.set_entry_point("init_context")

    def _route(graph_state: Dict[str, Any]) -> str:
        role = graph_state.get("role")
        return "agent" if role == "agent" else "user"

    graph.add_conditional_edges(
        "init_context",
        _route,
        {"agent": "agent_strategy", "user": "risk_gate"},
    )
    graph.add_edge("agent_strategy", "output")

    def _post_risk(graph_state: Dict[str, Any]) -> str:
        result = graph_state.get("risk_result") or {}
        if result.get("risk_level") == "high":
            return "output"
        return "strategy_flow"

    graph.add_conditional_edges(
        "risk_gate",
        _post_risk,
        {"output": "output", "strategy_flow": "strategy_flow"},
    )
    graph.add_edge("strategy_flow", "output")
    graph.add_edge("output", END)

    return graph


def build_main_graph(template_root: str) -> Any:
    """编译后的 LangGraph 主图。"""

    return create_state_graph(template_root).compile()


class GraphRuntime:
    """LangGraph 调用器，封装编译后的图"""

    def __init__(self, template_root: str) -> None:
        self.graph = build_main_graph(template_root)

    def invoke(self, session: SessionState, role: str, text: str | None = None) -> Dict[str, Any]:
        initial_state: Dict[str, Any] = {"session": session, "role": role, "text": text, "payload": {}}
        final_state = self.graph.invoke(initial_state)
        return final_state.get("payload", {})
