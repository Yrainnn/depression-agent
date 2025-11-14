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
        ctx = state.item_contexts.get(item_id)
        question = (
            item.get("question")
            or item.get("item_name")
            or (ctx.item_name if ctx and ctx.item_name else None)
            or _question_for(f"H{item_id:02d}")
        )
        score_raw = item.get("score")
        try:
            score_value = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score_value = None
        max_score = item.get("max_score") if isinstance(item.get("max_score"), (int, float)) else MAX_SCORE.get(item_id)
        entry: Dict[str, Any] = {
            "item_id": item_id,
            "question": question,
            "score": score_value,
            "max_score": max_score,
        }

        reason = item.get("reason")
        if isinstance(reason, str) and reason.strip():
            entry["reason"] = reason.strip()

        prepared.append(entry)
    return prepared


def prepare_report_payload(state: SessionState) -> Optional[Dict[str, Any]]:
    analysis = state.analysis or {}
    total_block = analysis.get("total_score") or {}
    per_item_source = analysis.get("per_item_scores") or total_block.get("items") or []
    items = per_item_source if isinstance(per_item_source, list) else []
    if not isinstance(items, list) or not items:
        return None

    per_item_scores = _prepare_per_item_scores(state, items)
    if not per_item_scores:
        return None

    payload: Dict[str, Any] = {
        "per_item_scores": per_item_scores,
        "total_score": total_block.get("sum") if isinstance(total_block, dict) else total_block,
        "max_total": total_block.get("max") if isinstance(total_block, dict) else analysis.get("max_total"),
        "diagnosis": analysis.get("diagnosis"),
        "advice": analysis.get("advice"),
    }

    total_value = payload.get("total_score")
    if isinstance(total_value, (int, float)):
        payload["total_score"] = float(total_value)
    else:
        computed_total = 0.0
        for entry in per_item_scores:
            score = entry.get("score")
            if isinstance(score, (int, float)):
                computed_total += float(score)
        payload["total_score"] = round(computed_total, 2)

    if not isinstance(payload.get("max_total"), (int, float)):
        payload["max_total"] = sum(MAX_SCORE.values())

    summary = state.patient_context.conversation_summary
    if isinstance(summary, str) and summary.strip():
        payload["summary"] = summary.strip()

    rationale = analysis.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale_lines = [
            f"H{entry['item_id']:02d}: {entry['reason']}"
            for entry in per_item_scores
            if entry.get("reason")
        ]
        rationale = "\n".join(rationale_lines) if rationale_lines else ""
    if rationale:
        payload["rationale"] = rationale

    return payload


__all__ = ["prepare_report_payload"]

