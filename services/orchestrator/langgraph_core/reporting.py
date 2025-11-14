from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.orchestrator.questions_hamd17 import MAX_SCORE

from .state_types import SessionState


def _normalize_item_id(raw_id: Any) -> Optional[int]:
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, str):
        stripped = raw_id.strip()
        if not stripped:
            return None
        if stripped.startswith("H") and stripped[1:].isdigit():
            return int(stripped[1:])
        if stripped.isdigit():
            return int(stripped)
    return None


def _prepare_per_item_scores(state: SessionState, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for item in items:
        item_id = _normalize_item_id(item.get("item_id"))
        if item_id is None:
            continue
        raw = item.get("raw") or {}
        ctx = state.item_contexts.get(item_id)
        entry: Dict[str, Any] = {
            "item_id": item_id,
            "question": (ctx.item_name if ctx and ctx.item_name else None),
            "score": item.get("score"),
            "max_score": raw.get("max_score") if isinstance(raw.get("max_score"), (int, float)) else MAX_SCORE.get(item_id),
        }

        confidence = raw.get("confidence")
        if isinstance(confidence, (int, float)):
            entry["confidence"] = confidence

        evidence_refs = raw.get("evidence_refs")
        if isinstance(evidence_refs, list):
            entry["evidence_refs"] = [str(ref) for ref in evidence_refs if ref]

        prepared.append(entry)
    return prepared


def prepare_report_payload(state: SessionState) -> Optional[Dict[str, Any]]:
    analysis = state.analysis or {}
    total_block = analysis.get("total_score") or {}
    items = total_block.get("items") or []
    if not isinstance(items, list) or not items:
        return None

    per_item_scores = _prepare_per_item_scores(state, items)
    if not per_item_scores:
        return None

    rationale_lines: List[str] = []
    for item in items:
        item_id = _normalize_item_id(item.get("item_id"))
        if item_id is None:
            continue
        raw = item.get("raw") or {}
        reason = raw.get("reason")
        if isinstance(reason, str) and reason.strip():
            rationale_lines.append(f"H{item_id:02d}: {reason.strip()}")

    payload: Dict[str, Any] = {
        "per_item_scores": per_item_scores,
        "items": per_item_scores,
        "total_score": total_block.get("sum"),
    }

    summary = state.patient_context.conversation_summary
    if isinstance(summary, str) and summary.strip():
        payload["summary"] = summary.strip()

    if rationale_lines:
        payload["opinion"] = {"rationale": "\n".join(rationale_lines)}

    return payload


__all__ = ["prepare_report_payload"]

