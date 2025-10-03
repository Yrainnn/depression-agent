from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from services.orchestrator.langgraph_min import orchestrator
from services.report.build import build_pdf
from services.store.repository import repository

router = APIRouter()

_SID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


async def _rate_limit() -> None:
    """Apply a minimal rate limiter for debug endpoints."""

    await asyncio.sleep(0.1)


def _validate_sid(sid: str) -> None:
    if not _SID_PATTERN.match(sid):
        raise HTTPException(status_code=400, detail="invalid session id")


class StepRequest(BaseModel):
    sid: str = Field(..., description="Conversation session identifier")
    text: Optional[str] = Field(None, description="Patient utterance text")
    audio_ref: Optional[str] = Field(None, description="Audio reference (path or URL)")


    class Config:
        allow_population_by_field_name = True


class StepResponse(BaseModel):
    next_utterance: str
    progress: Dict[str, Any]
    risk_flag: bool
    tts_text: Optional[str] = None
    tts_url: Optional[str] = None


@router.post("/dm/step", response_model=StepResponse)
async def dm_step(payload: StepRequest) -> StepResponse:
    if not payload.text and not payload.audio_ref:
        state = repository.load_session_state(payload.sid)
        transcripts = repository.load_transcripts(payload.sid)
        if not state and not transcripts:
            result = orchestrator.ask(payload.sid)
            return StepResponse(**result)
        raise HTTPException(status_code=400, detail="text or audio_ref must be provided")

    audio_ref = payload.audio_ref
    if audio_ref and audio_ref.startswith("file://"):
        audio_ref = audio_ref[7:]

    result = orchestrator.step(
        payload.sid,
        text=payload.text,
        audio_ref=audio_ref,
    )
    return StepResponse(**result)


@router.post("/upload/audio")
async def upload_audio(
    sid: str = Form(..., description="Conversation session identifier"),
    file: UploadFile = File(..., description="Audio file to upload"),
) -> Dict[str, Any]:
    _validate_sid(sid)

    upload_root = Path("/tmp/tingwu_uploads")
    dest_dir = upload_root / sid
    dest_dir.mkdir(parents=True, exist_ok=True)

    original_suffix = Path(file.filename or "").suffix
    suffix = original_suffix if original_suffix else ".wav"
    dest_path = dest_dir / f"{uuid4().hex}{suffix}"

    data = await file.read()
    dest_path.write_bytes(data)

    return {"audio_ref": f"file://{dest_path}"}


@router.get("/debug/session/{sid}")
async def debug_session(sid: str) -> Dict[str, Any]:
    _validate_sid(sid)
    await _rate_limit()

    state = repository.load_session_state(sid)
    transcripts = repository.load_transcripts(sid)
    scores = repository.load_scores(sid)

    return {
        "sid": sid,
        "state": state,
        "transcripts_count": len(transcripts),
        "score_exists": bool(scores),
    }


@router.get("/debug/risk/{sid}")
async def debug_risk(sid: str) -> Dict[str, Any]:
    _validate_sid(sid)
    await _rate_limit()

    items = repository.get_risk_recent(sid, count=20)

    return {
        "sid": sid,
        "count": len(items),
        "items": [
            {
                "id": item.get("id"),
                "ts": item.get("ts"),
                "reason": item.get("reason"),
                "match_text": item.get("match_text"),
            }
            for item in items
        ],
    }


class ReportRequest(BaseModel):
    sid: str

    class Config:
        allow_population_by_field_name = True


class ReportResponse(BaseModel):
    report_url: str


@router.post("/report/build", response_model=ReportResponse)
async def report_build(payload: ReportRequest) -> ReportResponse:
    state = repository.load_session_state(payload.sid)
    scores_payload = repository.load_scores(payload.sid)
    summary = state.get("summary")
    score_json: Dict[str, Any]
    if isinstance(scores_payload, dict):
        score_json = dict(scores_payload)
    else:
        score_json = {"per_item_scores": list(scores_payload or [])}

    if summary:
        score_json.setdefault("summary", summary)
        if isinstance(score_json.get("opinion"), dict):
            score_json["opinion"].setdefault("summary", summary)
        elif summary:
            score_json["opinion"] = {"summary": summary}

    report_payload = build_pdf(payload.sid, score_json)
    return ReportResponse(**report_payload)
