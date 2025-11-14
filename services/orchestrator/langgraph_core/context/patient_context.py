from __future__ import annotations

from typing import Iterable

from ..state_types import ItemContext, PatientContext

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


def reinforce_with_context(patient: PatientContext, item_ctx: ItemContext) -> None:
    """Persist the current patient snapshot into a specific item context."""

    if not item_ctx:
        return

    snapshot = patient.snapshot_for_item()
    item_ctx.absorb_patient_snapshot(snapshot)


def integrate_item_into_patient(patient: PatientContext, item_ctx: ItemContext) -> None:
    """Fold a finalized item record back into the short-term context."""

    if not item_ctx:
        return

    snapshot = item_ctx.export_for_patient()
    if snapshot:
        patient.integrate_item_snapshot(snapshot)


def refresh_patient_from_items(patient: PatientContext, items: Iterable[ItemContext]) -> None:
    """Rebuild patient context from all completed item contexts."""

    snapshots = [ctx.export_for_patient() for ctx in items if ctx]
    if not snapshots:
        return

    merged_summary_parts = []
    themes: list[str] = []
    facts: dict[str, object] = {}
    risks: list[str] = []

    for snapshot in snapshots:
        summary = snapshot.get("summary")
        if isinstance(summary, str) and summary.strip():
            merged_summary_parts.append(summary.strip())

        for theme in snapshot.get("themes", []) or []:
            if isinstance(theme, str) and theme and theme not in themes:
                themes.append(theme)

        for key, value in (snapshot.get("facts") or {}).items():
            facts[key] = value

        for risk in snapshot.get("risks", []) or []:
            if isinstance(risk, str) and risk and risk not in risks:
                risks.append(risk)

    if merged_summary_parts:
        patient.conversation_summary = " / ".join(merged_summary_parts)[-600:]

    if themes:
        patient.narrative_themes = themes

    if facts:
        patient.structured_facts.update(facts)

    if risks:
        patient.active_risks = risks
