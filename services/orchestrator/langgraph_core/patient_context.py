from __future__ import annotations

from typing import Dict, Iterable, List

from .state_types import ItemContext, PatientContext

_FUZZY = {"偶尔", "有点", "不太", "说不清", "还好", "一般"}
_DESPAIR = {"没意思", "绝望", "无望", "活着没意义"}
_SLEEP = {"凌晨", "早醒", "三四点", "五点醒"}


def update_from_answer(context: PatientContext, answer: str) -> PatientContext:
    """Update the short-term context using a new user answer."""

    if any(token in answer for token in _FUZZY):
        context.pending_clarifications.append(answer)

    if any(token in answer for token in _DESPAIR):
        if "绝望" not in context.narrative_themes:
            context.narrative_themes.append("绝望")

    if any(token in answer for token in _SLEEP):
        if "睡眠异常" not in context.narrative_themes:
            context.narrative_themes.append("睡眠异常")

    clipped = (context.conversation_summary + " " + answer).strip()
    context.conversation_summary = clipped[-500:]
    return context


def reinforce_with_context(
    context: PatientContext, item_contexts: Dict[int, ItemContext]
) -> PatientContext:
    """Feed the long-term summaries back into the short-term context."""

    seen: List[str] = []
    for item in item_contexts.values():
        for theme in item.themes:
            if theme not in seen:
                seen.append(theme)
    context.narrative_themes = seen + [t for t in context.narrative_themes if t not in seen]

    for item in item_contexts.values():
        context.structured_facts.update(item.facts)

    summaries: Iterable[str] = (item.summary for item in item_contexts.values() if item.summary)
    joined = " / ".join(summaries)
    if joined:
        context.conversation_summary = joined[-500:]
    return context
