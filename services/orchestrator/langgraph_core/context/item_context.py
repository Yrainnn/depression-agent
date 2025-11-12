from __future__ import annotations

from typing import Dict

from ..state_types import ItemContext, SessionState


def ensure_item_context(state: SessionState) -> ItemContext:
    if state.index not in state.item_contexts:
        state.item_contexts[state.index] = ItemContext(item_id=state.index, item_name=state.current_item_name)
    return state.item_contexts[state.index]


def append_dialogue(state: SessionState, role: str, text: str) -> None:
    ctx = ensure_item_context(state)
    ctx.dialogue.append({"role": role, "text": text})


def finalize_item_context(state: SessionState) -> Dict[str, object]:
    ctx = state.item_contexts.get(state.index)
    if not ctx:
        return {"item_saved": False}
    ctx.summary = state.patient_context.conversation_summary[-400:]
    ctx.themes = list(state.patient_context.narrative_themes)
    ctx.facts = dict(state.patient_context.structured_facts)
    return {"item_saved": True, "item_id": ctx.item_id, "name": ctx.item_name}
