from __future__ import annotations

from typing import Any, Dict, List

from ..context.item_context import append_dialogue, ensure_item_context
from ..llm_tools import LLM
from ..state_types import SessionState
from .base_node import Node


class StrategyNode(Node):
    """策略节点：读取模板并生成下一问"""

    def __init__(self, name: str):
        super().__init__(name)

    def _get_strategy_cfg(self, state: SessionState) -> Dict[str, Any]:
        template = state.current_template or {}
        strategies: List[Dict[str, Any]] = template.get("strategies", []) or []
        strategy_map = {cfg.get("id"): cfg for cfg in strategies if isinstance(cfg, dict)}
        current = state.current_strategy or "S2"
        if current not in strategy_map and strategy_map:
            current = next(iter(strategy_map))
            state.current_strategy = current
        cfg = strategy_map.get(current, {})
        state.current_branches = cfg.get("branches", []) or []
        return cfg

    def run(self, state: SessionState, **kwargs: Any) -> Dict[str, Any]:
        role = kwargs.get("role", "agent")
        if role == "agent":
            cfg = self._get_strategy_cfg(state)
            template = cfg.get("template", "请结合上下文提出问题。")
            ensure_item_context(state)
            payload = {
                "context": state.patient_context.to_prompt_snippet(),
                "template": template,
                "dialogue": state.item_contexts[state.index].dialogue,
                "progress": {"index": state.index, "total": state.total},
            }
            result = LLM.call("generate", payload) or {}
            question = result.get("text") or template
            question = str(question).strip()
            append_dialogue(state, "agent", question)
            state.last_agent_text = question
            state.waiting_for_user = True
            return {"ask": question, "strategy": state.current_strategy}
        return {}
