from __future__ import annotations

import logging
from typing import List, Optional

from packages.common.config import settings


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


class SDKTingWuASR:
    """Placeholder for the former DashScope TingWu integration."""

    def __init__(self, app_settings):  # pragma: no cover - configuration guard
        self.model = app_settings.TINGWU_MODEL or "paraformer-realtime-v2"
        self.api_key = app_settings.DASHSCOPE_API_KEY
        self.app_id = app_settings.TINGWU_APP_ID
        raise AsrError(
            "DashScope TingWu SDK support has been removed from this deployment"
        )

    def transcribe(
        self,
        text: Optional[str] = None,
        audio_ref: Optional[str] = None,
    ) -> List[dict]:  # pragma: no cover - configuration guard
        raise AsrError(
            "DashScope TingWu SDK support has been removed from this deployment"
        )


DEFAULT_ASR = StubASR()
_SDK_TINGWU_ASR: Optional[SDKTingWuASR] = None


def _provider() -> StubASR | SDKTingWuASR:
    global _SDK_TINGWU_ASR
    if settings.ASR_PROVIDER.lower() == "tingwu" and settings.DASHSCOPE_API_KEY:
        if _SDK_TINGWU_ASR is None:
            try:
                _SDK_TINGWU_ASR = SDKTingWuASR(settings)
            except Exception as exc:  # pragma: no cover - configuration guard
                LOGGER.warning("Failed to initialise SDKTingWuASR: %s", exc)
                return DEFAULT_ASR
        return _SDK_TINGWU_ASR
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


TingWuASR = SDKTingWuASR

__all__ = [
    "AsrError",
    "StubASR",
    "SDKTingWuASR",
    "TingWuASR",
    "transcribe",
]
