from __future__ import annotations

from typing import Any, Dict

from ..context.item_context import append_dialogue, ensure_item_context
from ..context.patient_context import reinforce_with_context, update_patient_context
from ..llm_tools import LLM
from ..state_types import SessionState
from .base_node import Node


class UpdateNode(Node):
    """用户回答后的上下文更新节点"""

    def __init__(self, name: str):
        super().__init__(name)

    def run(self, state: SessionState, **kwargs: Any) -> Dict[str, Any]:
        user_text = str(kwargs.get("user_text") or "")
        branch = kwargs.get("branch")
        next_strategy = kwargs.get("next_strategy") or state.current_strategy
        if isinstance(next_strategy, str):
            next_strategy = next_strategy.strip()
        if not user_text:
            if next_strategy and isinstance(next_strategy, str):
                state.current_strategy = next_strategy
                if next_strategy == "END":
                    state.completed = True
                    state.waiting_for_user = False
            return {"branch": branch, "next_strategy": next_strategy}

        ensure_item_context(state)
        append_dialogue(state, "user", user_text)
        state.last_user_text = user_text
        update_patient_context(state.patient_context, user_text)

        facts_resp = LLM.call("extract_facts", {"text": user_text}) or {}
        facts = facts_resp.get("facts") if isinstance(facts_resp, dict) else None
        if isinstance(facts, dict):
            state.patient_context.structured_facts.update(facts)

        themes_resp = LLM.call("identify_themes", {"text": user_text}) or {}
        themes = themes_resp.get("themes") if isinstance(themes_resp, dict) else None
        if isinstance(themes, list):
            for theme in themes:
                if theme and theme not in state.patient_context.narrative_themes:
                    state.patient_context.narrative_themes.append(theme)

        summary_resp = LLM.call(
            "summarize_context",
            {"prev": state.patient_context.conversation_summary, "new": user_text, "limit": 500},
        )
        summary = summary_resp.get("summary") if isinstance(summary_resp, dict) else None
        if isinstance(summary, str) and summary:
            state.patient_context.conversation_summary = summary

        reinforce_with_context(state.patient_context, state.item_contexts[state.index])

        state.waiting_for_user = False
        if isinstance(next_strategy, str) and next_strategy:
            state.current_strategy = next_strategy
        if state.current_strategy == "END":
            state.completed = True
        return {"branch": branch, "next_strategy": next_strategy, "completed": state.completed}
