from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from packages.common.config import settings
from services.audio.tingwu_client import transcribe as tingwu_playback_transcribe


LOGGER = logging.getLogger(__name__)


class AsrError(RuntimeError):
    """Raised when an ASR provider cannot fulfil a request."""


class StubASR:
    """Fallback ASR implementation that echoes supplied text."""

    def transcribe(
        self,
        text: Optional[str] = None,
        audio_ref: Optional[str] = None,
    ) -> List[dict]:
        if text:
            return [
                {
                    "utt_id": "u1",
                    "text": text,
                    "speaker": "patient",
                    "ts": [0, 0],
                    "conf": 0.95,
                }
            ]

        LOGGER.debug("StubASR invoked for audio_ref=%s", audio_ref)
        return []


class TingwuClientASR:
    """Adapter wrapping the internal TingWu playback client."""

    def __init__(self, app_settings):
        self._settings = app_settings
        self._validate_environment()

    def _validate_environment(self) -> None:
        missing: List[str] = []
        if not self._settings.ALIBABA_CLOUD_ACCESS_KEY_ID:
            missing.append("ALIBABA_CLOUD_ACCESS_KEY_ID")
        if not self._settings.ALIBABA_CLOUD_ACCESS_KEY_SECRET:
            missing.append("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
        if not (
            self._settings.TINGWU_APPKEY
            or self._settings.ALIBABA_TINGWU_APPKEY
        ):
            missing.append("ALIBABA_TINGWU_APPKEY")

        if missing:
            raise AsrError(
                "Missing TingWu configuration: " + ", ".join(sorted(set(missing)))
            )

    @staticmethod
    def _resolve_audio_path(audio_ref: str) -> Path:
        local_path = audio_ref
        if audio_ref.startswith("file://"):
            local_path = audio_ref.replace("file://", "", 1)
        path = Path(local_path)
        if not path.exists():
            raise AsrError(f"audio file not found: {path}")
        if not path.is_file():
            raise AsrError(f"audio path is not a file: {path}")
        return path

    def transcribe(
        self,
        text: Optional[str] = None,
        audio_ref: Optional[str] = None,
    ) -> List[dict]:
        if text:
            return DEFAULT_ASR.transcribe(text=text)
        if not audio_ref:
            raise AsrError("audio_ref required for TingWu transcription")

        path = self._resolve_audio_path(audio_ref)
        try:
            transcript = tingwu_playback_transcribe(str(path))
        except EnvironmentError as exc:
            raise AsrError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - network/SDK errors
            raise AsrError(f"TingWu transcription failed: {exc}") from exc

        transcript = (transcript or "").strip()
        if not transcript:
            LOGGER.debug("TingWu transcription returned empty result for %s", path)
            return []

        return [
            {
                "utt_id": "tw_1",
                "text": transcript,
                "speaker": "patient",
                "ts": [0, 0],
                "conf": 0.95,
            }
        ]


DEFAULT_ASR = StubASR()
_TINGWU_ASR: Optional[TingwuClientASR] = None
_TINGWU_INIT_FAILED = False


def _provider() -> StubASR | TingwuClientASR:
    global _TINGWU_ASR, _TINGWU_INIT_FAILED
    if _TINGWU_ASR is not None:
        return _TINGWU_ASR
    if _TINGWU_INIT_FAILED:
        return DEFAULT_ASR
    try:
        _TINGWU_ASR = TingwuClientASR(settings)
        return _TINGWU_ASR
    except AsrError as exc:
        LOGGER.warning(
            "Failed to initialise TingWu client ASR, falling back to stub: %s",
            exc,
        )
        _TINGWU_INIT_FAILED = True
        return DEFAULT_ASR


def transcribe(
    text: Optional[str] = None,
    audio_ref: Optional[str] = None,
) -> List[dict]:
    provider = _provider()
    try:
        return provider.transcribe(text=text, audio_ref=audio_ref)
    except AsrError as exc:
        LOGGER.warning("Primary ASR provider failed, falling back to stub: %s", exc)
        if provider is DEFAULT_ASR:
            return []
        return DEFAULT_ASR.transcribe(text=text, audio_ref=audio_ref)


TingWuASR = TingwuClientASR

__all__ = [
    "AsrError",
    "StubASR",
    "TingwuClientASR",
    "TingWuASR",
    "transcribe",
]
