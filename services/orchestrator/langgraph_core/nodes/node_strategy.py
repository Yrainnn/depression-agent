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

    def _build_strategy_graph(self, state: SessionState, template: Dict[str, Any]) -> None:
        raw_strategies = template.get("strategies", []) or []
        state.strategy_sequence = []
        state.strategy_graph = {}
        state.strategy_map = {}

        for entry in raw_strategies:
            if not isinstance(entry, dict):
                continue
            strategy_id = entry.get("id")
            if not isinstance(strategy_id, str):
                continue
            strategy_id = strategy_id.strip()
            if not strategy_id:
                continue

            normalized = dict(entry)
            normalized["id"] = strategy_id

            branches: List[Dict[str, Any]] = []
            for branch in normalized.get("branches", []) or []:
                if not isinstance(branch, dict):
                    continue
                normalized_branch = dict(branch)
                condition = normalized_branch.get("condition")
                if isinstance(condition, str):
                    normalized_branch["condition"] = condition.strip()
                next_step = normalized_branch.get("next")
                if isinstance(next_step, str):
                    normalized_branch["next"] = next_step.strip()
                prompt = normalized_branch.get("next_prompt")
                if isinstance(prompt, str):
                    normalized_branch["next_prompt"] = prompt.strip()
                branches.append(normalized_branch)
            normalized["branches"] = branches

            next_step = normalized.get("next")
            if isinstance(next_step, str):
                normalized["next"] = next_step.strip()

            state.strategy_sequence.append(strategy_id)
            state.strategy_map[strategy_id] = normalized

        for strategy_id in state.strategy_map:
            state.strategy_graph[strategy_id] = []

    def _get_strategy_cfg(self, state: SessionState) -> Dict[str, Any]:
        template = state.current_template or {}
        if not state.strategy_map:
            self._build_strategy_graph(state, template)

        if state.current_strategy == "END":
            state.current_branches = []
            state.default_next_strategy = ""
            return {}

        if not state.strategy_sequence:
            state.current_branches = []
            state.default_next_strategy = ""
            if not state.current_strategy:
                state.current_strategy = "END"
                state.completed = True
            return {}

        current = state.current_strategy or ""
        if current not in state.strategy_map:
            current = state.strategy_sequence[0]
            state.current_strategy = current

        cfg = state.strategy_map.get(current, {})
        branches = cfg.get("branches", []) or []
        state.current_branches = branches
        next_step = cfg.get("next")
        state.default_next_strategy = next_step if isinstance(next_step, str) else ""
        return cfg

    def run(self, state: SessionState, **kwargs: Any) -> Dict[str, Any]:
        role = kwargs.get("role", "agent")
        if role == "agent":
            cfg = self._get_strategy_cfg(state)
            if not cfg:
                state.waiting_for_user = False
                if state.current_strategy == "END":
                    state.completed = True
                state.pending_strategy = ""
                return {
                    "ask": None,
                    "strategy": state.current_strategy,
                    "branches": [],
                    "strategy_graph": state.strategy_graph,
                    "default_next": state.default_next_strategy or None,
                }

            override = state.strategy_prompt_overrides.pop(state.current_strategy, None)
            template_text = override or cfg.get("template") or "请结合上下文提出问题。"
            template_text = str(template_text).strip() or "请结合上下文提出问题。"

            ensure_item_context(state)
            state.pending_strategy = state.current_strategy
            payload = {
                "context": state.patient_context.to_prompt_snippet(),
                "template": template_text,
                "dialogue": state.item_contexts[state.index].dialogue,
                "progress": {"index": state.index, "total": state.total},
            }
            result = LLM.call("generate", payload) or {}
            question = result.get("text") or template_text
            question = str(question).strip()
            if question:
                append_dialogue(state, "agent", question)
                state.last_agent_text = question
                state.waiting_for_user = True
            else:
                state.waiting_for_user = False
            response: Dict[str, Any] = {
                "ask": question or None,
                "strategy": state.current_strategy,
                "branches": state.current_branches,
                "strategy_graph": state.strategy_graph,
            }
            if state.default_next_strategy:
                response["default_next"] = state.default_next_strategy
            if override:
                response["prompt_override"] = override
            clarify_prompt = cfg.get("clarify_prompt") or cfg.get("clarify_template")
            if isinstance(clarify_prompt, str) and clarify_prompt.strip():
                response["clarify_prompt"] = clarify_prompt.strip()
            return response
        return {}
