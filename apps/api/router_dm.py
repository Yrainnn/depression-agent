from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.audio import asr_adapter
from services.orchestrator.langgraph_min import orchestrator
from services.report.build import build_report
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
    audio_ref: Optional[str] = Field(None, description="Reserved for future audio input")


    class Config:
        allow_population_by_field_name = True


class StepResponse(BaseModel):
    next_utterance: str
    progress: Dict[str, Any]
    risk: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None


@router.post("/dm/step", response_model=StepResponse)
async def dm_step(payload: StepRequest) -> StepResponse:
    if not payload.text and not payload.audio_ref:
        state = repository.load_session_state(payload.sid)
        transcripts = repository.load_transcripts(payload.sid)
        if not state and not transcripts:
            result = orchestrator.ask(payload.sid)
            return StepResponse(**result)
        raise HTTPException(status_code=400, detail="text or audio_ref must be provided")

    segments = []
    if payload.text:
        segments.extend(asr_adapter.transcribe(text=payload.text))
    if payload.audio_ref:
        segments.extend(asr_adapter.transcribe(audio_ref=payload.audio_ref))

    result = orchestrator.step(payload.sid, segments=segments)
    return StepResponse(**result)


@router.get("/debug/session/{sid}")
async def debug_session(sid: str) -> Dict[str, Any]:
    _validate_sid(sid)
    await _rate_limit()

    state = repository.load_session_state(sid)
    transcripts = repository.load_transcripts(sid)
    scores = repository.load_scores(sid)

    return {
        "state": state,
        "transcripts_count": len(transcripts),
        "score_present": bool(scores),
    }


@router.get("/debug/risk/{sid}")
async def debug_risk(sid: str) -> Dict[str, Any]:
    _validate_sid(sid)
    await _rate_limit()

    events = repository.load_risk_events(sid)
    recent_events = events[-20:]

    return {"events": recent_events}


class ReportRequest(BaseModel):
    sid: str

    class Config:
        allow_population_by_field_name = True


class ReportResponse(BaseModel):
    report_url: str


@router.post("/report/build", response_model=ReportResponse)
async def report_build(payload: ReportRequest) -> ReportResponse:
    state = repository.load_session_state(payload.sid)
    transcripts = repository.load_transcripts(payload.sid)
    scores_payload = repository.load_scores(payload.sid)
    if isinstance(scores_payload, dict):
        scores = scores_payload.get("per_item_scores", [])
        opinion = scores_payload.get("opinion")
    else:
        scores = scores_payload
        opinion = None
    summary = state.get("summary", "感谢配合，本次访谈总结如下。")
    if opinion and not summary:
        summary = opinion

    report_url = build_report(payload.sid, summary, scores, transcripts)
    return ReportResponse(report_url=report_url)
