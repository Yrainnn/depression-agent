from pathlib import Path

from services.orchestrator.langgraph_core.nodes.node_output import OutputNode
from services.orchestrator.langgraph_core.state_types import SessionState


class _DummyOSS:
    enabled = True

    def store_artifact(self, sid, category, path, metadata=None):
        return f"https://oss.example/{category}/{Path(path).name}"


class _DummyTTS:
    def __init__(self, path):
        self._path = path

    def synthesize(self, sid: str, text: str, voice=None):
        self._path.write_bytes(b"fake")
        return str(self._path)


def test_output_node_adds_final_message_when_completed(monkeypatch, tmp_path):
    state = SessionState(sid="demo")
    state.completed = True
    node = OutputNode("output")
    node._tts_adapter = _DummyTTS(tmp_path / "closing.wav")  # type: ignore[attr-defined]
    node._digital_human_enabled = False  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "services.orchestrator.langgraph_core.media.oss_client",
        _DummyOSS(),
    )

    result = node.run(state, payload={})

    assert result["final_message"] == "评估结束,感谢您的参与."
    assert result["report_generated"] is False
    media = result["final_message_media"]
    assert media["tts_text"] == "评估结束,感谢您的参与."
    assert media["tts_local_path"].endswith("closing.wav")
    assert media["tts_url"] == "https://oss.example/tts/audio/closing.wav"
    assert media["media_type"] == "audio"
    assert result["final_message_tts_url"] == "https://oss.example/tts/audio/closing.wav"
    assert result["final_message_media_type"] == "audio"


def test_output_node_surfaces_report_result(monkeypatch, tmp_path):
    state = SessionState(sid="demo")
    state.completed = True
    state.report_result = {
        "report_url": "file:///tmp/demo.pdf",
        "path": "/tmp/demo.pdf",
    }
    node = OutputNode("output")
    node._tts_adapter = _DummyTTS(tmp_path / "closing.wav")  # type: ignore[attr-defined]
    node._digital_human_enabled = False  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "services.orchestrator.langgraph_core.media.oss_client",
        _DummyOSS(),
    )

    result = node.run(state, payload={})

    assert result["report_generated"] is True
    assert result["report_url"] == "file:///tmp/demo.pdf"
    assert result["report"]["path"] == "/tmp/demo.pdf"
    assert result["final_message_media"]["tts_url"] == "https://oss.example/tts/audio/closing.wav"
    assert result["final_message_tts_url"] == "https://oss.example/tts/audio/closing.wav"
