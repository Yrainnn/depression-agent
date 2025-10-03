from __future__ import annotations

import logging
import os
from typing import List, Optional

from dashscope.multimodal.tingwu.tingwu_realtime import (
    TingWuRealtime,
    TingWuRealtimeCallback,
)

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
    """DashScope TingWu implementation used for realtime transcription."""

    def __init__(self, app_settings):
        self.api_key = app_settings.DASHSCOPE_API_KEY
        assert self.api_key, "DASHSCOPE_API_KEY is required"
        self.app_id = getattr(app_settings, "TINGWU_APP_ID", None)
        self.base_address = getattr(app_settings, "TINGWU_BASE_ADDRESS", None)
        self.audio_format = app_settings.TINGWU_FORMAT or "pcm"
        self.sample_rate = int(app_settings.TINGWU_SR or 16000)
        self.lang = app_settings.TINGWU_LANG or "cn"

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
        with open(local_path, "rb") as handle:
            return handle.read()

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

        audio_bytes = self._read_bytes(audio_ref)
        callback = self._Cb()
        try:
            tingwu = TingWuRealtime(
                model=None,
                audio_format=self.audio_format,
                sample_rate=self.sample_rate,
                app_id=self.app_id,
                base_address=self.base_address,
                api_key=self.api_key,
                callback=callback,
                max_end_silence=3000,
            )
            tingwu.start()
            tingwu.send_audio_data(audio_bytes)
            tingwu.stop()
        except Exception as exc:  # pragma: no cover - network/SDK errors
            raise AsrError(f"TingWu SDK transcription failed: {exc}") from exc

        return callback.segments


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
