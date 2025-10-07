import io
import wave
from pathlib import Path

from services.tts.tts_adapter import TTSAdapter


def _make_wav_bytes(duration_seconds: float = 0.1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        frames = int(16000 * duration_seconds)
        wav_file.writeframes(b"\x01\x00" * frames)
    return buffer.getvalue()


def test_tts_adapter_uses_stub_when_dashscope_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("services.tts.tts_adapter._SPEECH_SYNTHESIZER_CLS", None)
    adapter = TTSAdapter(out_dir=str(tmp_path))

    url = adapter.synthesize("sid123", "你好")
    assert url.startswith("file://")

    audio_path = Path(url[len("file://") :])
    assert audio_path.exists()

    with wave.open(str(audio_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000


def test_tts_adapter_invokes_injected_factory(tmp_path):
    calls = []

    def factory(model: str, voice: str, audio_format: str):
        class _FakeSynth:
            def call(self, text: str, **kwargs):
                calls.append({
                    "model": model,
                    "voice": voice,
                    "audio_format": audio_format,
                    "text": text,
                    "kwargs": kwargs,
                })
                return _make_wav_bytes()

            def get_last_request_id(self):
                return "req-123"

            def get_first_package_delay(self):
                return 42.0

        return _FakeSynth()

    adapter = TTSAdapter(out_dir=str(tmp_path), synthesizer_factory=factory)

    url = adapter.synthesize("sid456", "测试合成", voice="custom_voice")
    assert url.startswith("file://")
    audio_path = Path(url[len("file://") :])
    assert audio_path.exists()

    assert calls, "Expected synthesizer factory to be invoked"
    call = calls[0]
    assert call["voice"] == "custom_voice"
    assert call["audio_format"] == "wav"
    assert call["kwargs"].get("format") == "wav"

    with wave.open(str(audio_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() > 0
