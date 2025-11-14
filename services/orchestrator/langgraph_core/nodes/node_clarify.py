from __future__ import annotations

import re
from typing import Any, Dict, List

from ..llm_tools import LLM
from ..state_types import SessionState
from .base_node import Node

_POSITIVE = {"有", "经常", "总是", "大部分", "明显", "非常", "抑郁", "低落", "难过"}
_NEGATIVE = {"没有", "不", "未", "无", "偶尔", "说不清", "一般", "否认", "否定", "含糊"}
_OSCILLATION = {"早上", "晚上", "白天", "夜里", "早醒", "凌晨"}


def _contains(text: str, tokens: List[str]) -> bool:
    return any(token in text for token in tokens)


def _fallback_match(condition: str, answer: str) -> bool:
    lowered = condition.lower()
    if any(keyword in lowered for keyword in ("肯定", "存在", "明确", "抑郁", "阳性")):
        return _contains(answer, list(_POSITIVE)) and not _contains(answer, list(_NEGATIVE))
    if any(keyword in lowered for keyword in ("否", "含糊", "模糊", "不清", "拒绝")):
        return _contains(answer, list(_NEGATIVE))
    if any(keyword in lowered for keyword in ("昼夜", "波动", "早醒", "晚", "早上", "晚上")):
        return _contains(answer, list(_OSCILLATION))

    tokens = [token for token in re.split(r"[\s,;；、\/|]+", condition) if len(token.strip()) > 1]
    if not tokens:
        tokens = [condition]
    return _contains(answer, tokens)


def _match_condition(condition: str, answer: str) -> bool:
    condition = (condition or "").strip()
    if not condition:
        return False

    payload = {"answer": answer, "condition": condition}
    try:
        result = LLM.call("match_condition", payload) or {}
    except Exception:
        result = {}

    match_value = result.get("match") if isinstance(result, dict) else None
    if isinstance(match_value, bool):
        return match_value
    if isinstance(match_value, str):
        normalized = match_value.strip().lower()
        if normalized in {"true", "yes", "1", "match", "positive"}:
            return True
        if normalized in {"false", "no", "0", "mismatch", "negative"}:
            return False

    return _fallback_match(condition, answer)


class ClarifyNode(Node):
    """调用大模型进行分支澄清"""

    def __init__(self, name: str):
        super().__init__(name)

    def _append_edge(
        self,
        state: SessionState,
        source: str,
        target: str,
        *,
        label: str | None = None,
        prompt_override: str | None = None,
    ) -> None:
        if not source or not target:
            return
        edges = state.strategy_graph.setdefault(source, [])
        for existing in edges:
            if existing.get("to") == target:
                if label:
                    existing["condition"] = label
                if prompt_override:
                    existing["prompt_override"] = prompt_override
                return

        descriptor: Dict[str, str] = {"to": target}
        if label:
            descriptor["condition"] = label
        if prompt_override:
            descriptor["prompt_override"] = prompt_override
        edges.append(descriptor)
        state.strategy_graph.setdefault(target, [])

    def run(self, state: SessionState, **kwargs: Any) -> Dict[str, Any]:
        answer = str(kwargs.get("user_text") or "")
        branches = state.current_branches
        source_strategy = state.pending_strategy or state.current_strategy
        if not answer:
            return {"branch": None, "next_strategy": state.current_strategy}

        payload = {"answer": answer, "branches": branches}
        result = LLM.call("clarify_branch", payload) or {}
        matched = result.get("matched")
        next_strategy = result.get("next")
        reason = result.get("reason")
        matched_branch: Dict[str, Any] | None = None

        if matched:
            for branch in branches:
                condition = branch.get("condition")
                if isinstance(condition, str) and condition == matched:
                    matched_branch = branch
                    break
        if isinstance(next_strategy, str):
            next_strategy = next_strategy.strip() or None

        if not next_strategy and branches:
            # fallback规则匹配
            for branch in branches:
                condition = branch.get("condition", "")
                if _match_condition(condition, answer):
                    matched = condition
                    next_strategy = branch.get("next", state.current_strategy)
                    reason = "rule_fallback"
                    matched_branch = branch
                    break

        clarify_prompt: str | None = None
        if not next_strategy and not branches and state.default_next_strategy:
            next_strategy = state.default_next_strategy
            reason = reason or "default_next"

        if not next_strategy:
            if source_strategy:
                attempts = state.clarify_attempts.get(source_strategy, 0) + 1
                state.clarify_attempts[source_strategy] = attempts
                if attempts >= max(state.max_clarify_attempts, 1) and state.default_next_strategy:
                    next_strategy = state.default_next_strategy
                    reason = "clarify_limit"
            if not next_strategy:
                clarify_cfg = state.strategy_map.get(source_strategy or "", {})
                prompt = clarify_cfg.get("clarify_prompt") or clarify_cfg.get("clarify_template")
                if isinstance(prompt, str) and prompt.strip():
                    clarify_prompt = prompt.strip()
                    state.strategy_prompt_overrides[source_strategy or state.current_strategy] = clarify_prompt
                return {
                    "branch": matched,
                    "next_strategy": None,
                    "clarify_reason": reason or "clarify_retry",
                    "clarify_prompt": clarify_prompt,
                }

        if not next_strategy and state.default_next_strategy:
            next_strategy = state.default_next_strategy
            reason = reason or "default_next"

        prompt_override: str | None = None
        if matched_branch is None and next_strategy:
            for branch in branches:
                nxt = branch.get("next")
                if isinstance(nxt, str) and nxt.strip() == next_strategy:
                    matched_branch = branch
                    break
        if matched_branch:
            prompt = matched_branch.get("next_prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompt_override = prompt.strip()

        if next_strategy:
            if source_strategy:
                edge_label = matched or reason
                if state.default_next_strategy == next_strategy and (not edge_label or edge_label in {"无可用分支", "rule_fallback"}):
                    edge_label = "default_next"
                self._append_edge(
                    state,
                    source_strategy,
                    next_strategy,
                    label=edge_label,
                    prompt_override=prompt_override,
                )
                state.clarify_attempts.pop(source_strategy, None)
            state.current_strategy = next_strategy
            if prompt_override:
                state.strategy_prompt_overrides[next_strategy] = prompt_override
            if next_strategy == "END":
                state.completed = True
            state.branch_history.append(f"{matched or 'unknown'}->{next_strategy}")
        return {
            "branch": matched,
            "next_strategy": next_strategy,
            "clarify_reason": reason,
            "prompt_override": prompt_override,
        }
