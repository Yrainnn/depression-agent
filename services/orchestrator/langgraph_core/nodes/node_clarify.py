from __future__ import annotations

from typing import Any, Dict, List

from ..llm_tools import LLM
from ..state_types import SessionState
from .base_node import Node

_POSITIVE = {"有", "经常", "总是", "大部分", "明显", "非常"}
_NEGATIVE = {"没有", "不", "未", "无", "偶尔", "说不清", "一般"}
_OSCILLATION = {"早上", "晚上", "白天", "夜里", "早醒"}


def _contains(text: str, tokens: List[str]) -> bool:
    return any(token in text for token in tokens)


def _match_condition(condition: str, answer: str) -> bool:
    condition = condition.strip()
    if not condition:
        return False
    if condition in {"明确存在抑郁情绪", "肯定", "存在"}:
        return _contains(answer, list(_POSITIVE)) and not _contains(answer, list(_NEGATIVE)[:4])
    if condition in {"否定或含糊", "否定/含糊", "否定", "含糊"}:
        return _contains(answer, list(_NEGATIVE))
    if condition in {"提及昼夜波动", "波动", "日夜变化"}:
        return _contains(answer, list(_OSCILLATION))
    return False


class ClarifyNode(Node):
    """调用大模型进行分支澄清"""

    def __init__(self, name: str):
        super().__init__(name)

    def run(self, state: SessionState, **kwargs: Any) -> Dict[str, Any]:
        answer = str(kwargs.get("user_text") or "")
        branches = state.current_branches
        if not answer:
            return {"branch": None, "next_strategy": state.current_strategy}

        payload = {"answer": answer, "branches": branches}
        result = LLM.call("clarify_branch", payload) or {}
        matched = result.get("matched")
        next_strategy = result.get("next")
        reason = result.get("reason")

        if not next_strategy and branches:
            # fallback规则匹配
            for branch in branches:
                condition = branch.get("condition", "")
                if _match_condition(condition, answer):
                    matched = condition
                    next_strategy = branch.get("next", state.current_strategy)
                    reason = "rule_fallback"
                    break
        if next_strategy:
            state.current_strategy = next_strategy
            state.branch_history.append(f"{matched or 'unknown'}->{next_strategy}")
        return {"branch": matched, "next_strategy": next_strategy, "clarify_reason": reason}
