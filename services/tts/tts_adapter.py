import logging
import os
import time
import uuid
import wave
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)
SR = 16000


class TTSAdapter:
    def __init__(self, out_dir: str = "/tmp/da_tts") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # 预留未来接入 CoSyVoice 的 Key
        self.api_key = os.getenv("DASHSCOPE_API_KEY")

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
        return f"file://{target.resolve()}"


def synthesize(text: str, *, voice: Optional[str] = None) -> str:
    adapter = TTSAdapter()
    return adapter.synthesize("default", text, voice)
