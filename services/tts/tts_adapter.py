import importlib
import importlib.util
import logging
import os
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from packages.common.config import settings
from services.oss.client import OSSClient, oss_client as default_oss_client

LOGGER = logging.getLogger(__name__)
SR = 16000


def _discover_dashscope() -> Tuple[Optional[Any], Optional[type]]:
    """Return the dashscope module and SpeechSynthesizer class if available."""

    if importlib.util.find_spec("dashscope") is None:
        return None, None

    dashscope_module = importlib.import_module("dashscope")
    if importlib.util.find_spec("dashscope.audio.tts_v2") is None:
        return dashscope_module, None

    tts_module = importlib.import_module("dashscope.audio.tts_v2")
    synthesizer_cls = getattr(tts_module, "SpeechSynthesizer", None)
    return dashscope_module, synthesizer_cls


_DASHSCOPE_MODULE, _SPEECH_SYNTHESIZER_CLS = _discover_dashscope()


class TTSAdapter:
    def __init__(
        self,
        out_dir: str = "/tmp/da_tts",
        *,
        synthesizer_factory: Optional[
            Callable[[str, str, Optional[str]], Any]
        ] = None,
        oss_client: Optional[OSSClient] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = (
            getattr(settings, "dashscope_api_key", None)
            or os.getenv("DASHSCOPE_API_KEY")
        )
        self.model = getattr(settings, "dashscope_tts_model", "cosyvoice-v2")
        self.voice = getattr(settings, "dashscope_tts_voice", "longxiaochun_v2")
        self.audio_format = (
            getattr(settings, "dashscope_tts_format", "wav") or "wav"
        ).lower()
        self.file_extension = self._normalize_extension(self.audio_format)

        self.oss_client = oss_client or default_oss_client

        if synthesizer_factory is not None:
            self._synthesizer_factory = synthesizer_factory
        elif _SPEECH_SYNTHESIZER_CLS and self.api_key:
            if _DASHSCOPE_MODULE is not None:
                _DASHSCOPE_MODULE.api_key = self.api_key

            def _factory(model: str, voice: str, audio_format: Optional[str]) -> Any:
                kwargs: Dict[str, Any] = {"model": model, "voice": voice}
                return _SPEECH_SYNTHESIZER_CLS(**kwargs)  # type: ignore[misc]

            self._synthesizer_factory = _factory
        else:
            self._synthesizer_factory = None

    def _normalize_extension(self, fmt: Optional[str]) -> str:
        if not fmt:
            return "wav"
        fmt_lower = fmt.lower().lstrip(".")
        if fmt_lower in {"wav", "wave"}:
            return "wav"
        if fmt_lower in {"mp3", "m4a", "aac"}:
            return fmt_lower
        return fmt_lower

    def _write_silence_wav(self, path: Path, seconds: float = 1.0) -> None:
        frames = int(SR * seconds)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SR)
            wav_file.writeframes(b"\x00\x00" * frames)

    def _call_synthesizer(
        self, synthesizer: Any, text: str, *, audio_format: Optional[str]
    ) -> Tuple[Optional[bytes], Optional[str], Optional[float]]:
        call_kwargs: Dict[str, Any] = {}
        if audio_format:
            call_kwargs["format"] = audio_format

        try:
            if call_kwargs:
                audio = synthesizer.call(text, **call_kwargs)
            else:
                audio = synthesizer.call(text)
        except TypeError:
            audio = synthesizer.call(text)

        request_id = None
        delay_ms = None
        if hasattr(synthesizer, "get_last_request_id"):
            try:
                request_id = synthesizer.get_last_request_id()
            except Exception:  # pragma: no cover - defensive log guard
                request_id = None
        if hasattr(synthesizer, "get_first_package_delay"):
            try:
                delay_ms = synthesizer.get_first_package_delay()
            except Exception:  # pragma: no cover - defensive log guard
                delay_ms = None

        return audio, request_id, delay_ms

    def synthesize(self, sid: str, text: str, voice: Optional[str] = None) -> str:
        session_dir = self.out_dir / sid
        session_dir.mkdir(parents=True, exist_ok=True)

        if text:
            text = text.strip()

        target_voice = voice or self.voice or ""

        if text and self._synthesizer_factory is not None:
            synthesizer: Optional[Any] = None
            try:
                synthesizer = self._synthesizer_factory(
                    self.model, target_voice, self.audio_format
                )
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.exception("Failed to initialise DashScope synthesizer")

            if synthesizer is not None:
                try:
                    audio_bytes, request_id, delay_ms = self._call_synthesizer(
                        synthesizer, text, audio_format=self.audio_format
                    )
                    if audio_bytes:
                        timestamp = int(time.time() * 1000)
                        filename = (
                            f"{sid}-{timestamp}-{uuid.uuid4().hex}.{self.file_extension}"
                        )
                        target = session_dir / filename
                        target.write_bytes(audio_bytes)
                        LOGGER.info(
                            "[TTS:dashscope] synthesized audio",
                            extra={
                                "sid": sid,
                                "path": str(target),
                                "voice": target_voice,
                                "text": text[:80] if text else "",
                                "request_id": request_id,
                                "first_package_delay_ms": delay_ms,
                            },
                        )
                        return self._finalize_audio(
                            sid,
                            target,
                            text,
                            target_voice,
                        )
                except Exception:
                    LOGGER.exception(
                        "DashScope TTS synthesis failed; falling back to stub"
                    )

        timestamp = int(time.time() * 1000)
        filename = f"{sid}-{timestamp}-{uuid.uuid4().hex}.wav"
        target = session_dir / filename
        self._write_silence_wav(target, seconds=1.0)
        LOGGER.info(
            "[TTS:stub] synthesized placeholder wav",
            extra={"sid": sid, "path": str(target), "voice": voice, "text": text[:80] if text else ""},
        )
        return self._finalize_audio(sid, target, text, target_voice)

    # ------------------------------------------------------------------
    def _finalize_audio(
        self,
        sid: str,
        path: Path,
        text: Optional[str],
        voice: Optional[str],
    ) -> str:
        url = f"file://{path.resolve()}"
        oss_url = self._upload_to_oss(sid, path, text=text, voice=voice)
        return oss_url or url

    def _upload_to_oss(
        self,
        sid: str,
        path: Path,
        *,
        text: Optional[str],
        voice: Optional[str],
    ) -> Optional[str]:
        client = self.oss_client
        if client is None or not getattr(client, "enabled", False):
            return None
        metadata = {
            "type": "tts",
            "voice": voice,
        }
        if text:
            metadata["text"] = text[:200]
        url = client.store_artifact(
            sid,
            "tts",
            path,
            metadata=metadata,
        )
        return url

    def _upload_to_oss(self, sid: str, file_path: Path) -> Optional[str]:
        if not self.uploader.enabled:
            self.last_upload = None
            return None

        base_prefix = self.oss_prefix.rstrip("/")
        prefix = f"{base_prefix}/{sid}/" if base_prefix else f"{sid}/"
        try:
            oss_key = self.uploader.upload_file(str(file_path), oss_key_prefix=prefix)
            url = self.uploader.get_presigned_url(oss_key, expires_minutes=24 * 60)
            self.last_upload = {"oss_key": oss_key, "url": url}
        except (OSError, OSSUploaderError) as exc:
            LOGGER.warning("Failed to upload TTS result for %s: %s", sid, exc)
            self.last_upload = None
            return None
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.exception("Unexpected error uploading TTS result for %s", sid)
            self.last_upload = None
            return None

        try:
            file_path.unlink()
            if not any(file_path.parent.iterdir()):
                file_path.parent.rmdir()
        except OSError:
            LOGGER.debug("Failed to clean up local TTS artefact %s", file_path, exc_info=True)

        return url


def synthesize(text: str, *, voice: Optional[str] = None) -> str:
    adapter = TTSAdapter()
    return adapter.synthesize("default", text, voice)
