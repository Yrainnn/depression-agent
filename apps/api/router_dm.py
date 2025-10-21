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
from services.orchestrator.langgraph_min import orchestrator
from services.oss.uploader import OSSUploader
from services.report.build import build_pdf
from services.tts.tts_adapter import TTSAdapter
from services.store.repository import repository

LOGGER = logging.getLogger(__name__)

router = APIRouter()

_SID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


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
    next_utterance: str
    progress: Dict[str, Any]
    risk_flag: bool
    tts_text: Optional[str] = None
    tts_url: Optional[str] = None
    video_url: Optional[str] = None
    media_type: Optional[str] = None
    segments_previews: Optional[List[str]] = None

    class Config:
        extra = "allow"


@router.post("/dm/step", response_model=StepResponse)
async def dm_step(payload: DMStepPayload) -> StepResponse:
    audio_ref = payload.audio_ref
    if audio_ref and audio_ref.startswith("file://"):
        audio_ref = audio_ref[7:]

    result = orchestrator.step(
        sid=payload.sid,
        role=payload.role,
        text=payload.text,
        audio_ref=audio_ref,
        scale=payload.scale or "HAMD17",
    )
    return StepResponse(**result)


@router.post("/dm/report")
async def generate_report(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
    except Exception:
        return {"error": "请求体格式错误"}

    sid = data.get("sid") if isinstance(data, dict) else None
    if not sid:
        return {"error": "缺少 sid 参数"}

    try:
        state = orchestrator._load_state(sid)
        score_payload = orchestrator._prepare_report_scores(sid, state)
    except Exception as exc:  # pragma: no cover - defensive guard
        return {"error": str(exc)}

    if not score_payload:
        return {"error": "无有效评分数据，无法生成报告"}

    try:
        report_result = build_pdf(sid, score_payload)
        if not isinstance(report_result, dict):
            return {"error": "报告文件生成失败"}

        report_url = report_result.get("report_url")
        local_path: Optional[str] = report_result.get("path") or report_result.get("file_path")

        uploader = OSSUploader()
        if uploader.enabled and local_path:
            try:
                oss_key = uploader.upload_file(str(local_path), oss_key_prefix="reports/")
                report_url = uploader.get_presigned_url(oss_key)
            except Exception as exc:  # pragma: no cover - network/service guard
                LOGGER.warning("OSS 上传失败，使用本地链接返回：%s", exc)

        if not report_url:
            if local_path:
                report_url = str(Path(str(local_path)).resolve().as_uri())
            else:
                return {"error": "报告链接生成失败"}
    except Exception as exc:
        return {"error": str(exc)}

    LOGGER.info("报告生成成功")
    return {"report_url": report_url, "sid": sid}


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
        segments = orchestrator.asr.transcribe(text=payload.text, audio_ref=audio_ref)
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
