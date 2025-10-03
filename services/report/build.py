from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from jinja2 import Environment, select_autoescape
from weasyprint import HTML

try:  # pragma: no cover - optional import during bootstrapping
    from services.store.repository import (
        ConversationRepository,
        repository as _shared_repository,
    )
except Exception:  # pragma: no cover - runtime guard
    ConversationRepository = None  # type: ignore
    _shared_repository = None  # type: ignore


LOGGER = logging.getLogger(__name__)

REPORT_VERSION = "v0.2"
REPORT_DIR = Path("/tmp/depression_agent_reports")


def _resolve_repository() -> Optional[ConversationRepository]:  # type: ignore[valid-type]
    repo = getattr(build_pdf, "_repository", None)
    if repo is not None:
        return repo
    if _shared_repository is not None:
        return _shared_repository
    if ConversationRepository is not None:  # pragma: no cover - fallback construction
        try:
            return ConversationRepository()
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to instantiate ConversationRepository")
    return None


def _normalize_per_item_scores(raw_scores: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in raw_scores or []:
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        try:
            score_value = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_value = None
        confidence = item.get("confidence") or item.get("confidence_score")
        evidence_refs = item.get("evidence_refs") or item.get("evidence") or []
        if isinstance(evidence_refs, str):
            try:
                evidence_refs = json.loads(evidence_refs)
            except json.JSONDecodeError:
                evidence_refs = [evidence_refs]
        if not isinstance(evidence_refs, list):
            evidence_refs = [str(evidence_refs)]
        normalized.append(
            {
                "item_id": item.get("item_id") or item.get("name") or "",
                "question": item.get("question") or item.get("name") or "",
                "score": score_value if score_value is not None else item.get("score", 0),
                "confidence": confidence,
                "evidence_refs": [str(ref) for ref in evidence_refs if ref],
            }
        )
    return normalized


def _compute_total_score(per_item_scores: List[Dict[str, Any]], score_json: Dict[str, Any]) -> float:
    total = score_json.get("total_score")
    if isinstance(total, (int, float)):
        return float(total)
    accum = 0.0
    for item in per_item_scores:
        try:
            accum += float(item.get("score", 0) or 0)
        except (TypeError, ValueError):
            continue
    return round(accum, 2)


def _render(template: str, context: Dict[str, Any]) -> str:
    env = Environment(
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.from_string(template).render(**context)


def _prepare_risk_events(repo: Optional[ConversationRepository], sid: str) -> List[Dict[str, Any]]:  # type: ignore[valid-type]
    if repo is None:
        return []

    events: List[Dict[str, Any]] = []
    try:
        if hasattr(repo, "get_risk_recent"):
            events = repo.get_risk_recent(sid, count=10)  # type: ignore[attr-defined]
    except Exception:
        LOGGER.exception("Failed to load risk events via stream for %s", sid)
        events = []

    if not events:
        try:
            events = repo.load_risk_events(sid)
        except Exception:
            LOGGER.exception("Failed to load risk events for %s", sid)
            events = []

    normalized: List[Dict[str, Any]] = []
    for raw in events[-10:]:
        if not isinstance(raw, dict):
            continue
        event = dict(raw)
        timestamp = event.get("timestamp") or event.get("ts") or event.get("time")
        if isinstance(timestamp, (list, tuple)):
            timestamp = ",".join(str(part) for part in timestamp)
        if not timestamp and event.get("stream_id"):
            timestamp = str(event["stream_id"])
        snippet = (
            event.get("text")
            or event.get("utterance")
            or event.get("segment")
            or event.get("excerpt")
            or event.get("snippet")
        )
        if isinstance(snippet, (list, tuple)):
            snippet = " ".join(str(part) for part in snippet if part)
        reason = event.get("reason")
        if not reason:
            triggers = event.get("triggers")
            if isinstance(triggers, (list, tuple)):
                reason = "、".join(str(trigger) for trigger in triggers if trigger)
            elif triggers:
                reason = str(triggers)
        level = event.get("level")
        reason_parts = [str(level)] if level else []
        if reason:
            reason_parts.append(str(reason))
        normalized.append(
            {
                "timestamp": timestamp or "—",
                "snippet": snippet or "（无文本）",
                "reason": " / ".join(reason_parts) if reason_parts else "",
            }
        )
    return normalized


def build_pdf(sid: str, score_json: Dict[str, Any]) -> Dict[str, str]:
    repo = _resolve_repository()
    per_item_scores = _normalize_per_item_scores(
        score_json.get("per_item_scores")
        or score_json.get("items")
        or score_json.get("scores")
        or []
    )

    summary = score_json.get("summary")
    opinion = score_json.get("opinion") if isinstance(score_json.get("opinion"), dict) else {}
    if not summary:
        summary = opinion.get("summary") or opinion.get("overall") or ""
    rationale = opinion.get("rationale") if isinstance(opinion, dict) else ""

    total_score = _compute_total_score(per_item_scores, score_json)

    risk_events = _prepare_risk_events(repo, sid)

    context = {
        "sid": sid,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "report_version": REPORT_VERSION,
        "summary": summary,
        "rationale": rationale,
        "total_score": total_score,
        "per_item_scores": per_item_scores,
        "risk_events": risk_events,
    }

    base_styles = """
    <style>
        body { font-family: 'Noto Sans', 'Helvetica', sans-serif; padding: 32px; color: #2c3e50; }
        h1 { font-size: 24px; margin-bottom: 4px; }
        h2 { font-size: 18px; margin-top: 24px; }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        th, td { border: 1px solid #d7d7d7; padding: 8px; font-size: 13px; }
        th { background-color: #f2f5f7; }
        .meta { color: #6c7a89; font-size: 12px; }
        .score-total { font-size: 32px; font-weight: bold; margin: 12px 0; }
        .muted { color: #7f8c8d; font-size: 12px; }
        ul { padding-left: 20px; }
        li { margin-bottom: 6px; }
    </style>
    """

    detailed_template = """
    <html>
    <head>
        <meta charset=\"utf-8\" />
        __BASE_STYLES__
    </head>
    <body>
        <h1>抑郁评估报告 {{ report_version }}</h1>
        <div class=\"meta\">会话 ID：{{ sid }} · 生成时间：{{ generated_at }}</div>
        {% if summary %}
        <div class=\"section\">
            <h2>摘要</h2>
            <p>{{ summary }}</p>
            {% if rationale %}<p class=\"muted\">说明：{{ rationale }}</p>{% endif %}
        </div>
        {% endif %}
        <div class=\"section\">
            <h2>总分</h2>
            <div class=\"score-total\">{{ total_score }} / 27</div>
        </div>
        {% if per_item_scores %}
        <div class=\"section\">
            <h2>分项分</h2>
            <table>
                <thead>
                    <tr><th>条目 ID</th><th>分数 (0-3)</th><th>置信度</th><th>证据引用</th></tr>
                </thead>
                <tbody>
                    {% for item in per_item_scores %}
                    <tr>
                        <td>{{ item.item_id or '—' }}</td>
                        <td>{{ item.score }}</td>
                        <td>{{ item.confidence or '—' }}</td>
                        <td>{{ item.evidence_refs | join(', ') if item.evidence_refs else '—' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% else %}
        <div class=\"section\">
            <h2>评估结果</h2>
            <p>数据不足，建议线下面谈。</p>
        </div>
        {% endif %}
        <div class=\"section\">
            <h2>风险事件摘要</h2>
            {% if risk_events %}
            <ul>
                {% for event in risk_events %}
                <li><strong>{{ event.timestamp }}</strong> · {{ event.snippet }}{% if event.reason %}（{{ event.reason }}）{% endif %}</li>
                {% endfor %}
            </ul>
            {% else %}
            <p>无风险事件记录。</p>
            {% endif %}
        </div>
    </body>
    </html>
    """.replace("__BASE_STYLES__", base_styles)

    html = _render(detailed_template, context)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / f"report_{sid}.pdf"
    HTML(string=html).write_pdf(str(output_path))
    return {"report_url": output_path.resolve().as_uri()}
