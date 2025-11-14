from __future__ import annotations

from typing import Any, Dict

from ..llm_tools import LLM
from ..state_types import SessionState
from .base_node import Node

class ClarifyNode(Node):
    """调用大模型进行分支澄清"""

    def __init__(self, name: str):
        super().__init__(name)

    def run(self, state: SessionState, **kwargs: Any) -> Dict[str, Any]:
        answer = str(kwargs.get("user_text") or "")
        branches = state.current_branches
        if not answer:
            return {
                "branch": None,
                "next_strategy": state.current_strategy,
                "clarify": False,
            }

        template_cfg = {}
        strategies = (state.current_template or {}).get("strategies", []) or []
        for cfg in strategies:
            if cfg.get("id") == state.current_strategy:
                template_cfg = cfg
                break

        payload = {
            "answer": answer,
            "branches": branches,
            "context": state.patient_context.to_prompt_snippet(),
            "template": template_cfg.get("template", ""),
        }
        result = LLM.call("clarify_branch", payload) or {}

        matched = result.get("matched")
        next_strategy = result.get("next")
        reason = result.get("reason")
        clarify_flag = bool(result.get("clarify"))
        clarify_question = result.get("clarify_question")

        if clarify_flag and not clarify_question:
            extra = LLM.call(
                "clarify_question",
                {
                    "context": state.patient_context.to_prompt_snippet(),
                    "template": payload.get("template", ""),
                    "answer": answer,
                },
            )
            clarify_question = (extra or {}).get("question")

        if next_strategy:
            state.current_strategy = next_strategy
            state.branch_history.append(f"{matched or 'unknown'}->{next_strategy}")

        return {
            "branch": matched,
            "next_strategy": next_strategy,
            "clarify_reason": reason,
            "clarify": clarify_flag,
            "clarify_question": clarify_question,
        }
