from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from services.orchestrator.prompts import get_prompt

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
        state.patient_context = update_from_answer(state.patient_context, answer)

        facts = LLM.extract_facts(answer)
        if facts:
            state.patient_context.structured_facts.update(facts)

        item_ctx = state.item_contexts.setdefault(
            state.index, ItemContext(item_id=state.index, item_name=state.current_item_name)
        )
        item_ctx.dialogue.append({"role": "user", "text": answer})

        strategy_map = {entry["id"]: entry for entry in template.get("strategies", [])}
        current = state.current_strategy or "S2"
        current_cfg = strategy_map.get(current) or next(iter(strategy_map.values()), {})

        for branch in current_cfg.get("branches", []):
            condition = branch.get("condition", "")
            if not condition:
                continue
            if _match_condition(condition, answer):
                next_strategy = branch.get("next")
                state.current_strategy = next_strategy or current
                state.strategy_history.append(f"{current}->{next_strategy}({condition})")
                hint = branch.get("next_prompt")
                if hint:
                    state.patient_context.pending_clarifications.append(hint)
                state.strategy_substep_idx = 0
                payload: Dict[str, Any] = {"branch": condition, "next_strategy": state.current_strategy}
                if hint:
                    payload["hint"] = hint
                return payload

        return {"branch": None}

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

        prompt = get_prompt("question_generation").format(
            context=f"{state.patient_context.to_prompt_snippet()}{supplement}",
            strategy_id=current,
            strategy_name=cfg.get("name", ""),
            template=question_template,
        )

        question = LLM.generate(prompt)
        state.last_role = "agent"
        state.last_agent_text = question
        state.waiting_for_user = True

        item_ctx = state.item_contexts.setdefault(
            state.index, ItemContext(item_id=state.index, item_name=state.current_item_name)
        )
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
