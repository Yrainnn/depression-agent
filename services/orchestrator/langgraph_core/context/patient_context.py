from __future__ import annotations

from typing import Dict

from ..state_types import PatientContext, ItemContext

_FUZZY_TOKENS = ("偶尔", "一般", "不太", "说不清", "还好")
_THEME_TOKENS = {"绝望": ("绝望", "没意思", "无望", "活不下去"), "睡眠异常": ("凌晨", "早醒", "三四点", "五点醒")}


def update_patient_context(context: PatientContext, answer: str) -> None:
    """Update short-term context with the latest user answer."""
    normalized = answer.strip()
    if not normalized:
        return
    if context.conversation_summary:
        summary = f"{context.conversation_summary} {normalized}".strip()
    else:
        summary = normalized
    context.conversation_summary = summary[-600:]

    for theme, tokens in _THEME_TOKENS.items():
        if any(token in normalized for token in tokens) and theme not in context.narrative_themes:
            context.narrative_themes.append(theme)

    if any(token in normalized for token in _FUZZY_TOKENS) and "含糊" not in context.narrative_themes:
        context.narrative_themes.append("含糊")


def reinforce_patient_context(context: PatientContext, item_contexts: Dict[int, ItemContext]) -> None:
    """Merge long-term insights back into the short-term context between items."""
    if not item_contexts:
        return
    merged_summary = " / ".join(
        ic.summary for ic in item_contexts.values() if ic.summary
    )[-600:]
    if merged_summary:
        context.conversation_summary = merged_summary

    merged_themes: list[str] = []
    for ic in item_contexts.values():
        for theme in ic.themes:
            if theme not in merged_themes:
                merged_themes.append(theme)
    if merged_themes:
        context.narrative_themes = merged_themes

    for ic in item_contexts.values():
        if ic.facts:
            context.structured_facts.update(ic.facts)
