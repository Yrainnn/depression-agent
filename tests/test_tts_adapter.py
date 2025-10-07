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


def test_tts_adapter_invokes_injected_factory_for_each_call(tmp_path):
    calls = []
    creations = []
    instances = []

    def factory(model: str, voice: str, audio_format: str):
        creations.append((model, voice, audio_format))

        class _FakeSynth:
            def __init__(self) -> None:
                self.call_count = 0

            def call(self, text: str, **kwargs):
                self.call_count += 1
                calls.append({
                    "model": model,
                    "voice": voice,
                    "audio_format": audio_format,
                    "text": text,
                    "kwargs": kwargs,
                    "call_count": self.call_count,
                })
                return _make_wav_bytes()

            def get_last_request_id(self):
                return "req-123"

            def get_first_package_delay(self):
                return 42.0

        synth = _FakeSynth()
        instances.append(synth)
        return synth

    adapter = TTSAdapter(out_dir=str(tmp_path), synthesizer_factory=factory)

    url1 = adapter.synthesize("sid456", "测试合成", voice="custom_voice")
    url2 = adapter.synthesize("sid456", "再次合成", voice="custom_voice")

    for url in (url1, url2):
        assert url.startswith("file://")
        audio_path = Path(url[len("file://") :])
        assert audio_path.exists()
        with wave.open(str(audio_path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getnframes() > 0

    assert creations == [
        ("cosyvoice-v2", "custom_voice", "wav"),
        ("cosyvoice-v2", "custom_voice", "wav"),
    ]
    assert len(instances) == 2
    assert calls and len(calls) == 2
    assert calls[0]["voice"] == "custom_voice"
    assert calls[0]["kwargs"].get("format") == "wav"
    assert calls[0]["call_count"] == 1
    assert calls[1]["call_count"] == 1


def test_tts_adapter_uploads_audio_when_oss_enabled(tmp_path):
    uploads = []

    class FakeOSS:
        enabled = True

        def store_artifact(self, sid, category, path, metadata=None):
            uploads.append({
                "sid": sid,
                "category": category,
                "path": str(path),
                "metadata": metadata,
            })
            filename = Path(path).name
            return f"https://oss.example/{category}/{filename}"

    def factory(model: str, voice: str, audio_format: str):
        class _FakeSynth:
            def call(self, text: str, **kwargs):
                return _make_wav_bytes()

        return _FakeSynth()

    adapter = TTSAdapter(
        out_dir=str(tmp_path),
        synthesizer_factory=factory,
        oss_client=FakeOSS(),
    )

    url = adapter.synthesize("oss-sid", "需要上传")

    assert url.startswith("https://oss.example/tts/")
    assert uploads and uploads[0]["category"] == "tts"
    assert Path(uploads[0]["path"]).exists()
    assert uploads[0]["metadata"]["type"] == "tts"
