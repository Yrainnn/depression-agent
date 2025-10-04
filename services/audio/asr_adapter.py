from __future__ import annotations

import logging
import os
from typing import List, Optional

from packages.common.config import settings
from services.audio import tingwu_client


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
    """Adapter wrapping the local Tingwu client implementation."""

    def __init__(self, _app_settings=settings):  # pragma: no cover - simple init
        self._settings = _app_settings

    def _resolve_path(self, path: str) -> str:
        local_path = path.replace("file://", "", 1) if path.startswith("file://") else path
        if not os.path.exists(local_path):
            raise AsrError(f"audio file not found: {local_path}")
        return local_path

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

        if not audio_ref:
            raise AsrError("audio_ref required")

        local_path = self._resolve_path(audio_ref)

        try:
            transcript = tingwu_client.transcribe(local_path)
        except EnvironmentError as exc:  # pragma: no cover - configuration guard
            raise AsrError(f"Tingwu client misconfigured: {exc}") from exc
        except Exception as exc:  # pragma: no cover - network/SDK errors
            raise AsrError(f"Tingwu client transcription failed: {exc}") from exc

        transcript = transcript.strip()
        if not transcript:
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
_TINGWU_CLIENT_ASR: Optional[TingwuClientASR] = None


def _provider() -> StubASR | TingwuClientASR:
    global _TINGWU_CLIENT_ASR
    if _TINGWU_CLIENT_ASR is None:
        try:
            _TINGWU_CLIENT_ASR = TingwuClientASR(settings)
        except Exception as exc:  # pragma: no cover - configuration guard
            LOGGER.warning("Failed to initialise TingwuClientASR: %s", exc)
            return DEFAULT_ASR
    return _TINGWU_CLIENT_ASR


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
