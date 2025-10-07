import sys
import types
from typing import Any, Dict

import pytest

_jinja2 = types.ModuleType("jinja2")


class _FakeTemplate:
    def render(self, **_: Any) -> str:
        return ""


class _FakeEnv:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def from_string(self, _template: str) -> _FakeTemplate:
        return _FakeTemplate()


_jinja2.Environment = lambda *args, **kwargs: _FakeEnv(*args, **kwargs)
_jinja2.select_autoescape = lambda *args, **kwargs: None
sys.modules.setdefault("jinja2", _jinja2)

_weasyprint = types.ModuleType("weasyprint")


class _FakeHTML:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def write_pdf(self, _path: str) -> None:
        return None


_weasyprint.HTML = lambda *args, **kwargs: _FakeHTML(*args, **kwargs)
sys.modules.setdefault("weasyprint", _weasyprint)

from services.orchestrator.langgraph_min import LangGraphMini, SessionState


@pytest.fixture(autouse=True)
def _stub_pick_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("services.orchestrator.langgraph_min.pick_primary", lambda item_id: "下一题？")


def _make_state() -> SessionState:
    state = SessionState(sid="sid")
    state.index = 5
    state.total = 17
    state.scores_acc = [
        {
            "item_id": "H05",
            "score": 2,
            "symptom_summary": "测试",
            "dialogue_evidence": "引用",
            "evidence_refs": ["u1"],
            "score_type": "类型1",
            "score_reason": "原因",
        }
    ]
    state.analysis = {
        "items": [],
        "total_score": {
            "得分序列": "1,0",
            "pre_correction_total": 1,
            "corrected_total": 1,
            "correction_basis": "",
        },
    }
    state.opinion = "当前情绪评分较低，如有需要仍可与专业人士交流。"
    return state


def test_report_request_generates_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = LangGraphMini.__new__(LangGraphMini)
    orchestrator._persist_state = lambda state: None  # type: ignore[assignment]

    captured: Dict[str, Any] = {}

    def fake_prepare(self: LangGraphMini, sid: str, state: SessionState) -> Dict[str, Any]:
        captured["prepared"] = True
        return {
            "items": [
                {
                    "item_id": "H05",
                    "score": 2,
                }
            ]
        }

    def fake_build_pdf(sid: str, payload: Dict[str, Any]) -> Dict[str, str]:
        captured["payload"] = payload
        return {"report_url": "file:///tmp/report_sid.pdf"}

    def fake_make_response(
        self: LangGraphMini,
        sid: str,
        state: SessionState,
        text: str,
        *,
        turn_type: str,
        extra: Dict[str, Any] | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        response = {"next_utterance": text, "turn_type": turn_type}
        if extra:
            response.update(extra)
        return response

    monkeypatch.setattr(LangGraphMini, "_prepare_report_scores", fake_prepare, raising=False)
    monkeypatch.setattr("services.orchestrator.langgraph_min.build_pdf", fake_build_pdf)
    monkeypatch.setattr(LangGraphMini, "_make_response", fake_make_response, raising=False)

    state = _make_state()

    result = orchestrator._maybe_handle_report_request("sid", state, "我想获取报告")

    assert result is not None
    assert result["report_generated"] is True
    assert result["report_url"] == "file:///tmp/report_sid.pdf"
    assert "报告" in result["next_utterance"]
    assert captured["payload"]["items"][0]["score"] == 2


def test_report_request_without_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = LangGraphMini.__new__(LangGraphMini)
    orchestrator._persist_state = lambda state: None  # type: ignore[assignment]

    def fake_prepare(self: LangGraphMini, sid: str, state: SessionState) -> None:
        return None

    def fake_make_response(
        self: LangGraphMini,
        sid: str,
        state: SessionState,
        text: str,
        *,
        turn_type: str,
        extra: Dict[str, Any] | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        response = {"next_utterance": text, "turn_type": turn_type}
        if extra:
            response.update(extra)
        return response

    monkeypatch.setattr(LangGraphMini, "_prepare_report_scores", fake_prepare, raising=False)
    monkeypatch.setattr(LangGraphMini, "_make_response", fake_make_response, raising=False)

    state = _make_state()

    result = orchestrator._maybe_handle_report_request("sid", state, "我想生成报告")

    assert result is not None
    assert result.get("report_generated") is False
    assert "继续评估" in result["next_utterance"]


def test_non_report_text_returns_none() -> None:
    orchestrator = LangGraphMini.__new__(LangGraphMini)
    state = _make_state()
    assert orchestrator._maybe_handle_report_request("sid", state, "好的") is None
