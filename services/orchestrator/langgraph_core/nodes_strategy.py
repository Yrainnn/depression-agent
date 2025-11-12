from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .llm_tools import LLM
from .patient_context import update_from_answer
from .state_types import ItemContext, SessionState
from .yaml_loader import load_yaml_file

_POSITIVE = ("有", "经常", "总是", "大部分时间", "明显", "很", "非常", "持续")
_NEGATIVE = ("没有", "不", "未", "无", "并不", "不太", "很少", "偶尔", "说不清", "一般", "还行")
_OSCILLATION = ("早上", "晚上", "白天", "夜里", "早醒", "清晨")


def _contains(text: str, needles: List[str]) -> bool:
    return any(needle in text for needle in needles)


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


class TemplateProvider:
    """Load YAML templates that define each item's strategy pipeline."""

    def __init__(self, root: str) -> None:
        self.root = root

    def get_item(self, item_id: int) -> Dict[str, Any]:
        path = os.path.join(self.root, f"item_{item_id:02d}.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return load_yaml_file(path)


class StrategyNode:
    """Execute YAML-driven strategies and produce the next prompt."""

    def __init__(self, template_root: str) -> None:
        self.provider = TemplateProvider(template_root)

    def run(self, state: SessionState, user_text: Optional[str]) -> Dict[str, Any]:
        template = self.provider.get_item(state.index)
        state.current_item_name = template.get("project_name", f"item_{state.index}")

        if state.waiting_for_user and user_text is not None:
            return self._handle_user_answer(state, template, user_text)

        return self._ask_next_question(state, template)

    # ------------------------------------------------------------------
    def _handle_user_answer(
        self, state: SessionState, template: Dict[str, Any], answer: str
    ) -> Dict[str, Any]:
        state.last_role = "user"
        state.last_user_text = answer
        state.waiting_for_user = False
        previous_summary = state.patient_context.conversation_summary
        state.patient_context = update_from_answer(state.patient_context, answer)

        summary_result = LLM.call(
            "summarize_context",
            {"prev": previous_summary, "new": answer, "limit": 500},
        )
        summary = summary_result.get("summary") if isinstance(summary_result, dict) else None
        if isinstance(summary, str) and summary.strip():
            state.patient_context.conversation_summary = summary.strip()

        theme_result = LLM.call("identify_themes", {"text": answer})
        themes: List[str] = []
        if isinstance(theme_result, dict):
            raw_themes = theme_result.get("themes")
            if isinstance(raw_themes, list):
                themes = [t.strip() for t in raw_themes if isinstance(t, str) and t.strip()]
        for theme in themes:
            if theme not in state.patient_context.narrative_themes:
                state.patient_context.narrative_themes.append(theme)

        fact_result = LLM.call("extract_facts", {"text": answer})
        facts: Dict[str, Any] = {}
        if isinstance(fact_result, dict):
            raw_facts = fact_result.get("facts")
            if isinstance(raw_facts, dict):
                facts = {k: v for k, v in raw_facts.items() if v not in (None, "")}
        if facts:
            state.patient_context.structured_facts.update(facts)

        item_ctx = state.item_contexts.setdefault(
            state.index, ItemContext(item_id=state.index, item_name=state.current_item_name)
        )
        item_ctx.dialogue.append({"role": "user", "text": answer})
        if facts:
            item_ctx.facts.update(facts)
        for theme in themes:
            if theme not in item_ctx.themes:
                item_ctx.themes.append(theme)

        strategy_map = {entry["id"]: entry for entry in template.get("strategies", [])}
        current = state.current_strategy or "S2"
        current_cfg = strategy_map.get(current) or next(iter(strategy_map.values()), {})

        branches = current_cfg.get("branches", [])
        selected_branch: Optional[Dict[str, Any]] = None
        clarify_reason: Optional[str] = None
        if branches:
            clarify_result = LLM.call(
                "clarify_branch",
                {"answer": answer, "branches": branches, "strategy_id": current},
            )
            if isinstance(clarify_result, dict):
                matched = clarify_result.get("matched")
                next_strategy = clarify_result.get("next")
                reason = clarify_result.get("reason")
                if isinstance(reason, str) and reason.strip():
                    clarify_reason = reason.strip()
                if next_strategy:
                    for branch in branches:
                        if branch.get("next") == next_strategy and (
                            not matched or branch.get("condition") == matched
                        ):
                            selected_branch = branch
                            break
                if selected_branch is None and matched:
                    for branch in branches:
                        if branch.get("condition") == matched:
                            selected_branch = branch
                            break

        if selected_branch is None:
            for branch in branches:
                condition = branch.get("condition", "")
                if condition and _match_condition(condition, answer):
                    selected_branch = branch
                    if clarify_reason is None:
                        clarify_reason = "heuristic_match"
                    break

        if selected_branch:
            next_strategy = selected_branch.get("next")
            condition = selected_branch.get("condition")
            state.current_strategy = next_strategy or current
            state.strategy_history.append(f"{current}->{next_strategy}({condition})")
            hint = selected_branch.get("next_prompt")
            if hint:
                state.patient_context.pending_clarifications.append(hint)
            state.strategy_substep_idx = 0
            payload: Dict[str, Any] = {
                "branch": condition,
                "next_strategy": state.current_strategy,
            }
            if hint:
                payload["hint"] = hint
            if clarify_reason:
                payload["clarify_reason"] = clarify_reason
            return payload

        return {"branch": None, "clarify_reason": clarify_reason}

    def _ask_next_question(self, state: SessionState, template: Dict[str, Any]) -> Dict[str, Any]:
        strategy_map = {entry["id"]: entry for entry in template.get("strategies", [])}
        current = state.current_strategy or "S2"
        if current not in strategy_map and strategy_map:
            current = next(iter(strategy_map.keys()))
        state.current_strategy = current
        cfg = strategy_map.get(current, {})

        question_template = cfg.get("template") or "请给出该策略下的一句提问。"
        cluster_prompts: List[str] = (
            cfg.get("cluster_prompts")
            or cfg.get("cluster_steps")
            or ([] if not cfg.get("cluster") else [question_template])
        )
        if cluster_prompts:
            idx = state.strategy_substep_idx
            if idx < len(cluster_prompts):
                question_template = cluster_prompts[idx]
                state.strategy_substep_idx += 1
            else:
                state.strategy_substep_idx = 0
                question_template = cfg.get("template", question_template)

        hints = "\n".join(state.patient_context.pending_clarifications)
        if hints:
            state.patient_context.pending_clarifications.clear()
            supplement = f"\n【补充提示】\n{hints}"
        else:
            supplement = ""

        item_ctx = state.item_contexts.setdefault(
            state.index, ItemContext(item_id=state.index, item_name=state.current_item_name)
        )
        dialogue_window = item_ctx.dialogue[-6:]
        progress = {
            "item_id": state.index,
            "strategy": current,
            "turn_index": len(item_ctx.dialogue),
        }

        generation_payload = {
            "context": f"{state.patient_context.to_prompt_snippet()}{supplement}",
            "template": question_template,
            "strategy_id": current,
            "strategy_name": cfg.get("name", ""),
            "dialogue": dialogue_window,
            "progress": progress,
        }
        generated = LLM.call("generate", generation_payload)
        question = None
        if isinstance(generated, dict):
            question = generated.get("text") or generated.get("question")
        if not isinstance(question, str) or not question.strip():
            question = question_template
        state.last_role = "agent"
        state.last_agent_text = question
        state.waiting_for_user = True
        item_ctx.dialogue.append({"role": "agent", "text": question})

        return {"ask": question, "strategy": current}

    # ------------------------------------------------------------------
    def finalize_item(self, state: SessionState) -> Dict[str, Any]:
        context = state.item_contexts.get(state.index)
        if not context:
            return {}
        context.summary = state.patient_context.conversation_summary[-300:]
        context.themes = list(state.patient_context.narrative_themes)
        context.facts = dict(state.patient_context.structured_facts)

        state.current_strategy = ""
        state.strategy_substep_idx = 0
        state.waiting_for_user = False

        return {"item_saved": True, "item_id": state.index, "name": state.current_item_name}
