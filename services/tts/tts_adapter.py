import logging
import os
import time
import uuid
import wave
from pathlib import Path
from typing import Dict, Optional

from services.oss import OSSUploader, OSSUploaderError

LOGGER = logging.getLogger(__name__)
SR = 16000


class TTSAdapter:
    def __init__(
        self,
        out_dir: str = "/tmp/da_tts",
        *,
        uploader: Optional[OSSUploader] = None,
        oss_prefix: str = "tts/",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # 预留未来接入 CoSyVoice 的 Key
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.uploader = uploader or OSSUploader()
        self.oss_prefix = oss_prefix
        self.last_upload: Optional[Dict[str, str]] = None

    def _write_silence_wav(self, path: Path, seconds: float = 1.0) -> None:
        frames = int(SR * seconds)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SR)
            wav.writeframes(b"\x00\x00" * frames)

    def synthesize(self, sid: str, text: str, voice: Optional[str] = None) -> str:
        timestamp = int(time.time() * 1000)
        filename = f"{sid}-{timestamp}-{uuid.uuid4().hex}.wav"
        session_dir = self.out_dir / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        target = session_dir / filename
        self._write_silence_wav(target, seconds=1.0)
        LOGGER.info(
            "[TTS:stub] synthesized placeholder wav",
            extra={"sid": sid, "path": str(target), "voice": voice, "text": text[:80]},
        )

        oss_url = self._upload_to_oss(sid, target)
        if oss_url:
            return oss_url
        return f"file://{target.resolve()}"

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
