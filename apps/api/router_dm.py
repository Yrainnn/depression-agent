from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from packages.common.config import settings
from services.audio import asr_adapter
from services.orchestrator.config.item_registry import ITEM_IDS
from services.orchestrator.langgraph_main import LangGraphCoordinator
from services.orchestrator.langgraph_core.reporting import prepare_report_payload
from services.report.build import build_pdf
from services.tts.tts_adapter import TTSAdapter
from services.store.repository import repository

LOGGER = logging.getLogger(__name__)

router = APIRouter()

_SID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

_REGISTRY_LOCK = asyncio.Lock()
_SESSION_LOCKS: Dict[str, asyncio.Lock] = {}
_COORDINATORS: Dict[str, LangGraphCoordinator] = {}
_TOTAL_ITEMS = len(ITEM_IDS) or 17


async def _get_session_lock(sid: str) -> asyncio.Lock:
    async with _REGISTRY_LOCK:
        lock = _SESSION_LOCKS.get(sid)
        if lock is None:
            lock = asyncio.Lock()
            _SESSION_LOCKS[sid] = lock
        return lock


async def _get_coordinator(sid: str) -> LangGraphCoordinator:
    async with _REGISTRY_LOCK:
        coordinator = _COORDINATORS.get(sid)
        if coordinator is None:
            coordinator = LangGraphCoordinator(total_items=_TOTAL_ITEMS, sid=sid)
            _COORDINATORS[sid] = coordinator
        return coordinator


async def _rate_limit() -> None:
    """Apply a minimal rate limiter for debug endpoints."""

    await asyncio.sleep(0.1)


def _validate_sid(sid: str) -> None:
    if not _SID_PATTERN.match(sid):
        raise HTTPException(status_code=400, detail="invalid session id")


class DMStepPayload(BaseModel):
    sid: str
    role: Literal["user", "assistant"] = "user"
    text: Optional[str] = None
    audio_ref: Optional[str] = None
    scale: Optional[str] = "HAMD17"


class StepResponse(BaseModel):
    ts: str
    sid: str
    progress: Dict[str, Any]
    waiting_for_user: bool
    current_item: Dict[str, Any]
    current_strategy: Optional[str] = None
    patient_context: Optional[Dict[str, Any]] = None
    ask: Optional[str] = None
    media: Optional[Dict[str, Any]] = None
    tts_url: Optional[str] = None
    video_url: Optional[str] = None
    media_type: Optional[str] = None
    risk_level: Optional[str] = None
    risk_result: Optional[Dict[str, Any]] = None
    risk_media: Optional[Dict[str, Any]] = None
    final_message: Optional[str] = None
    final_message_media: Optional[Dict[str, Any]] = None
    final_message_tts_url: Optional[str] = None
    final_message_video_url: Optional[str] = None
    report_generated: Optional[bool] = None
    report_url: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    completed: Optional[bool] = None
    risk_flag: Optional[bool] = None
    next_utterance: Optional[str] = None

    class Config:
        extra = "allow"

@router.post("/dm/step", response_model=StepResponse)
async def dm_step(payload: DMStepPayload) -> StepResponse:
    sid = payload.sid
    _validate_sid(sid)

    coordinator = await _get_coordinator(sid)
    lock = await _get_session_lock(sid)
    async with lock:
        combined: Dict[str, Any] = {}

        if payload.text:
            user_result = coordinator.step(role="user", text=payload.text)
            if isinstance(user_result, dict):
                combined.update(user_result)

        if not coordinator.state.completed:
            agent_result = coordinator.step(role="agent")
            if isinstance(agent_result, dict):
                combined.update(agent_result)

        combined.setdefault("sid", coordinator.state.sid)
        combined.setdefault(
            "progress",
            {"index": coordinator.state.index, "total": coordinator.state.total},
        )
        combined.setdefault("waiting_for_user", coordinator.state.waiting_for_user)
        combined.setdefault(
            "current_item",
            {"id": coordinator.state.index, "name": coordinator.state.current_item_name},
        )
        combined.setdefault(
            "patient_context",
            coordinator.state.patient_context.snapshot_for_item(),
        )
        combined["completed"] = coordinator.state.completed

        if "ask" in combined and not combined.get("next_utterance"):
            combined["next_utterance"] = combined["ask"]
        elif combined.get("final_message") and not combined.get("next_utterance"):
            combined["next_utterance"] = combined["final_message"]

        risk_source = combined.get("risk_result") if isinstance(combined.get("risk_result"), dict) else {}
        risk_level = str(
            combined.get("risk_level")
            or risk_source.get("risk_level")
            or ""
        ).lower()
        if risk_level:
            combined["risk_level"] = risk_level
        combined["risk_flag"] = risk_level == "high"

        return StepResponse(**combined)


@router.post("/dm/report")
async def generate_report(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
    except Exception:
        return {"error": "请求体格式错误"}

    sid = data.get("sid") if isinstance(data, dict) else None
    if not sid:
        return {"error": "缺少 sid 参数"}

    coordinator = await _get_coordinator(sid)
    lock = await _get_session_lock(sid)
    async with lock:
        if coordinator.state.analysis is None:
            coordinator.score_node.run(coordinator.state)

        if coordinator.state.report_payload is None:
            coordinator.state.report_payload = prepare_report_payload(coordinator.state)

        payload = coordinator.state.report_payload
        if not payload:
            return {"error": "无有效评分数据，无法生成报告"}

        if coordinator.state.report_result is None:
            try:
                coordinator.state.report_result = build_pdf(
                    coordinator.state.sid, dict(payload)
                )
            except Exception as exc:  # pragma: no cover - report guard
                LOGGER.error("报告生成失败：%s", exc)
                return {"error": str(exc)}

        report_result = coordinator.state.report_result or {}
        report_url = report_result.get("report_url")
        if not report_url:
            local_path = report_result.get("path") or report_result.get("file_path")
            if isinstance(local_path, str) and local_path:
                report_url = str(Path(local_path).resolve().as_uri())
            else:
                return {"error": "报告链接生成失败"}

        LOGGER.info("报告生成成功")
        return {"report_url": report_url, "sid": sid, "report": report_result}


class AsrTranscribeRequest(BaseModel):
    sid: str
    text: Optional[str] = None
    audio_ref: Optional[str] = None


class AsrTranscribeResponse(BaseModel):
    sid: str
    segments_count: int
    json_url: str


@router.post("/asr/transcribe", response_model=AsrTranscribeResponse)
async def asr_transcribe(payload: AsrTranscribeRequest) -> AsrTranscribeResponse:
    _validate_sid(payload.sid)

    audio_ref = payload.audio_ref
    if audio_ref and audio_ref.startswith("file://"):
        audio_ref = audio_ref[7:]

    try:
        segments = asr_adapter.transcribe(text=payload.text, audio_ref=audio_ref)
    except Exception as exc:  # pragma: no cover - passthrough errors to client
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    transcripts_dir = Path("/tmp/transcripts")
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcripts_dir / f"{payload.sid}.json"
    transcript_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    return AsrTranscribeResponse(
        sid=payload.sid,
        segments_count=len(segments),
        json_url=f"file://{transcript_path}",
    )




class TTSPayload(BaseModel):
    sid: str
    text: str
    voice: Optional[str] = None


@router.post("/tts/say")
async def tts_say(payload: TTSPayload) -> Dict[str, Any]:
    _validate_sid(payload.sid)
    tts = TTSAdapter()
    try:
        url = tts.synthesize(payload.sid, payload.text, payload.voice)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"tts_url": url}


@router.post("/upload/audio")
async def upload_audio(
    sid: str = Form(..., description="Conversation session identifier"),
    file: UploadFile = File(..., description="Audio file to upload"),
) -> Dict[str, Any]:
    _validate_sid(sid)

    upload_root = Path("/tmp/tingwu_uploads")
    dest_dir = upload_root / sid
    dest_dir.mkdir(parents=True, exist_ok=True)

    original_suffix = Path(file.filename or "").suffix or ".tmp"
    original_path = dest_dir / f"{uuid4().hex}{original_suffix}"
    data = await file.read()
    original_path.write_bytes(data)

    converted_path = dest_dir / f"{uuid4().hex}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(original_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(converted_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:  # pragma: no cover - environment specific
        raise HTTPException(status_code=500, detail="ffmpeg not available") from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail="audio conversion failed") from exc
    finally:
        try:
            if original_path.exists() and original_path != converted_path:
                original_path.unlink()
        except OSError:
            pass

    return {"audio_ref": f"file://{converted_path}"}


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


@router.get("/debug/asr")
async def debug_asr() -> Dict[str, Any]:
    await _rate_limit()

    return {
        "provider": "tingwu_client",
        "region": settings.TINGWU_REGION,
        "sample_rate": settings.TINGWU_SAMPLE_RATE,
        "format": settings.TINGWU_FORMAT,
        "appkey_present": bool(settings.TINGWU_APPKEY),
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
