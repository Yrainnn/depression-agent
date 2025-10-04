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


class SDKTingWuASR:
    """DashScope TingWu implementation used for realtime transcription."""

    def __init__(self, app_settings):
        self.model = getattr(app_settings, "TINGWU_MODEL", "paraformer-realtime-v2")
        self.api_key = getattr(app_settings, "DASHSCOPE_API_KEY", None)
        if not self.api_key:
            raise AsrError("DASHSCOPE_API_KEY is required for TingWu SDK")

        self.app_id = getattr(app_settings, "TINGWU_APP_ID", None)
        if not self.app_id:
            raise AsrError("TINGWU_APP_ID is required for TingWu SDK")

        self.base_address = getattr(app_settings, "TINGWU_BASE_ADDRESS", None)
        self.audio_format = getattr(app_settings, "TINGWU_FORMAT", "pcm")
        self.sample_rate = int(
            getattr(app_settings, "TINGWU_SR", None)
            or getattr(app_settings, "TINGWU_SAMPLE_RATE", 16000)
        )
        self.lang = getattr(app_settings, "TINGWU_LANG", "cn")

    class _Cb(TingWuRealtimeCallback):
        def __init__(self):
            self._inc = 0
            self.segments: List[dict] = []

        def on_recognize_result(self, result):  # type: ignore[override]
            transcription = (
                (result or {})
                .get("payload", {})
                .get("output", {})
                .get("transcription", {})
            )
            if transcription.get("sentenceEnd") is True:
                begin = transcription.get("beginTime") or 0
                end = transcription.get("endTime") or 0
                text = transcription.get("text") or ""
                self._inc += 1
                self.segments.append(
                    {
                        "utt_id": f"tw_{self._inc}",
                        "text": text,
                        "speaker": "patient",
                        "ts": [begin / 1000.0, end / 1000.0],
                        "conf": 0.95,
                    }
                )

        def on_error(self, error):  # type: ignore[override]
            LOGGER.debug("TingWu realtime error: %s", error)

        def on_close(self):  # type: ignore[override]
            LOGGER.debug("TingWu realtime closed")

        def on_speech_listen(self, result):  # type: ignore[override]
            LOGGER.debug("TingWu speech listen: %s", result)

        def on_stopped(self, result):  # type: ignore[override]
            LOGGER.debug("TingWu realtime stopped: %s", result)

    def _read_bytes(self, path: str) -> bytes:
        local_path = path.replace("file://", "") if path.startswith("file://") else path
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


def _provider() -> StubASR | SDKTingWuASR:
    global _SDK_TINGWU_ASR
    provider_name = getattr(settings, "ASR_PROVIDER", "").lower()
    api_key = getattr(settings, "DASHSCOPE_API_KEY", None)
    if provider_name == "tingwu" and api_key:
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


TingWuASR = TingwuClientASR

__all__ = [
    "AsrError",
    "StubASR",
    "TingwuClientASR",
    "TingWuASR",
    "transcribe",
]
