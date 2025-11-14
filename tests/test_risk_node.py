from pathlib import Path

from services.orchestrator.langgraph_core.nodes.node_risk import RiskNode
from services.orchestrator.langgraph_core.state_types import SessionState


class _DummyOSS:
    enabled = True

    def store_artifact(self, sid, category, path, metadata=None):
        return f"https://oss.example/{category}/{Path(path).name}"


class _DummyTTS:
    def __init__(self, path):
        self._path = path

    def synthesize(self, sid: str, text: str, voice=None):
        self._path.write_bytes(b"risk")
        return str(self._path)


def test_risk_node_attaches_media(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "services.orchestrator.langgraph_core.nodes.node_risk.LLM.call",
        lambda tool, payload: {
            "risk_level": "high",
            "message": "我们检测到高危内容，请立即联系专业人员。",
            "triggers": ["自伤"],
        },
    )

    monkeypatch.setattr(
        "services.orchestrator.langgraph_core.media.oss_client",
        _DummyOSS(),
    )

    node = RiskNode("risk")
    node._tts_adapter = _DummyTTS(tmp_path / "risk.wav")  # type: ignore[attr-defined]
    node._digital_human_enabled = False  # type: ignore[attr-defined]

    state = SessionState(sid="demo")
    result = node.run(state, user_text="我真的不想活了")

    assert result["risk_level"] == "high"
    assert result["message"].startswith("我们检测到")
    media = result["risk_media"]
    assert media["tts_text"].startswith("我们检测到")
    assert media["tts_local_path"].endswith("risk.wav")
    assert media["tts_url"] == "https://oss.example/tts/audio/risk.wav"
    assert media["media_type"] == "audio"
    assert result["risk_tts_url"] == "https://oss.example/tts/audio/risk.wav"
    assert result["risk_media_type"] == "audio"
