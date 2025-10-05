from __future__ import annotations

"""Development-only TingWu ASR helpers."""

import json
import subprocess
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from services.audio.tingwu_client import transcribe as tingwu_transcribe


router = APIRouter(prefix="/asr/tingwu", tags=["asr-tingwu-file"])

UPLOAD_ROOT = Path("/tmp/asr_uploads")
TRANSCRIPTS_ROOT = Path("/tmp/transcripts")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_ROOT.mkdir(parents=True, exist_ok=True)


class TranscribeRequest(BaseModel):
    sid: str
    audio_ref: str


@router.post("/upload")
async def upload_audio(sid: str = Form(...), file: UploadFile = File(...)) -> dict:
    target_dir = UPLOAD_ROOT / sid
    target_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "").suffix.lower() or ".pcm"
    filename = f"{uuid.uuid4().hex}{suffix}"
    destination = target_dir / filename

    contents = await file.read()
    destination.write_bytes(contents)

    return {"audio_ref": f"file://{destination}"}


def _ensure_pcm(path: Path) -> Path:
    """Ensure the audio file is PCM 16 kHz mono for TingWu realtime."""

    if path.suffix.lower() == ".pcm":
        return path

    converted = path.with_suffix(".pcm")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(converted),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:  # pragma: no cover - best effort conversion
        raise HTTPException(status_code=400, detail=f"ffmpeg 转码失败: {exc}") from exc

    return converted


@router.post("/transcribe")
async def transcribe(payload: TranscribeRequest) -> dict:
    if not payload.audio_ref.startswith("file://"):
        raise HTTPException(status_code=400, detail="audio_ref 仅支持 file://")

    source = Path(payload.audio_ref.replace("file://", ""))
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"audio file not found: {source}")

    pcm_path = _ensure_pcm(source)
    transcript_text = tingwu_transcribe(str(pcm_path))

    transcript_path = TRANSCRIPTS_ROOT / f"{payload.sid}.json"
    transcript_path.write_text(
        json.dumps({"sid": payload.sid, "text": transcript_text}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "sid": payload.sid,
        "text": transcript_text,
        "transcript": f"file://{transcript_path}",
    }
