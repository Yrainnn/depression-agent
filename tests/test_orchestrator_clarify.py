import sys
import types
from typing import Any, Dict, List, Optional, Tuple

_nls = types.ModuleType("nls")
setattr(_nls, "enableTrace", lambda *_args, **_kwargs: None)
setattr(_nls, "setLogFile", lambda *_args, **_kwargs: None)
sys.modules.setdefault("nls", _nls)

_aliyun = types.ModuleType("aliyunsdkcore")
_aliyun_client = types.ModuleType("aliyunsdkcore.client")
setattr(_aliyun_client, "AcsClient", object)
_aliyun.client = _aliyun_client
_aliyun_request = types.ModuleType("aliyunsdkcore.request")
setattr(_aliyun_request, "CommonRequest", object)
_aliyun.request = _aliyun_request
sys.modules.setdefault("aliyunsdkcore", _aliyun)
sys.modules.setdefault("aliyunsdkcore.client", _aliyun_client)
sys.modules.setdefault("aliyunsdkcore.request", _aliyun_request)

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

packages_stub = types.ModuleType("packages")
common_stub = types.ModuleType("packages.common")
config_stub = types.ModuleType("packages.common.config")

store_repo_stub = types.ModuleType("services.store.repository")


class _RepoStub:
    def load_session_state(self, sid: str) -> Dict[str, Any]:
        return {}

    def save_session_state(self, sid: str, payload: Dict[str, Any]) -> None:
        return None

    def append_transcript(self, sid: str, event: Dict[str, Any]) -> None:
        return None

    def get_transcripts(self, sid: str) -> List[Dict[str, Any]]:
        return []

    def clear_last_clarify_need(self, sid: str) -> None:
        return None

    def get_last_clarify_need(self, sid: str) -> Optional[Dict[str, Any]]:
        return None

    def set_last_clarify_need(self, sid: str, item_id: int, need: str) -> None:
        return None

    def merge_scores(self, sid: str, payload: Dict[str, Any]) -> None:
        return None

    def mark_finished(self, sid: str) -> None:
        return None

    def push_risk_event_stream(self, sid: str, payload: Dict[str, Any]) -> None:
        return None

    def push_risk_event(self, sid: str, payload: Dict[str, Any]) -> None:
        return None

    def append_risk_event(self, sid: str, payload: Dict[str, Any]) -> None:
        return None

    def save_scores(self, sid: str, payload: Dict[str, Any]) -> None:
        return None


store_repo_stub.repository = _RepoStub()
sys.modules["services.store.repository"] = store_repo_stub


class _Settings:
    def __init__(self) -> None:
        self.deepseek_api_base = None
        self.deepseek_api_key = None
        self.ENABLE_DS_CONTROLLER = False
        self.ALIBABA_CLOUD_ACCESS_KEY_ID = ""
        self.ALIBABA_CLOUD_ACCESS_KEY_SECRET = ""
        self.TINGWU_REGION = "cn-beijing"
        self.TINGWU_APPKEY = ""
        self.ALIBABA_TINGWU_APPKEY = ""
        self.TINGWU_AK_ID = ""
        self.TINGWU_AK_SECRET = ""
        self.TINGWU_BASE = "https://example"
        self.TINGWU_WS_BASE = "wss://example"
        self.TINGWU_SAMPLE_RATE = 16000
        self.TINGWU_FORMAT = "pcm"
        self.TINGWU_LANG = "cn"
        self.OSS_ENDPOINT = ""
        self.OSS_BUCKET = ""
        self.OSS_PREFIX = ""
        self.OSS_ACCESS_KEY_ID = ""
        self.OSS_ACCESS_KEY_SECRET = ""
        self.OSS_BASE_URL = ""


config_stub.settings = _Settings()
packages_stub.common = common_stub
common_stub.config = config_stub

sys.modules["packages"] = packages_stub
sys.modules["packages.common"] = common_stub
sys.modules["packages.common.config"] = config_stub

import pytest

from services.llm.json_client import HAMDItem, HAMDResult, HAMDTotal
from services.orchestrator.langgraph_min import LangGraphMini, SessionState


settings = config_stub.settings


class _DummyRepo:
    def __init__(self) -> None:
        self.session: Dict[str, Dict[str, Any]] = {}
        self.last_set: Optional[Tuple[str, int, str]] = None

    def save_session_state(self, sid: str, payload: Dict[str, Any]) -> None:
        self.session[sid] = dict(payload)

    def append_transcript(self, sid: str, event: Dict[str, Any]) -> None:  # pragma: no cover - not used
        self.session.setdefault(sid, {})

    def get_transcripts(self, sid: str) -> List[Dict[str, Any]]:
        return []

    def clear_last_clarify_need(self, sid: str) -> None:  # pragma: no cover - not used
        return None

    def get_last_clarify_need(self, sid: str) -> Optional[Dict[str, Any]]:
        return None

    def set_last_clarify_need(self, sid: str, item_id: int, need: str) -> None:  # pragma: no cover - not used
        self.last_set = (sid, item_id, need)

    def mark_finished(self, sid: str) -> None:  # pragma: no cover - not used
        return None

    def save_scores(self, sid: str, payload: Dict[str, Any]) -> None:  # pragma: no cover - not used
        return None


class _DummyDeepSeek:
    def usable(self) -> bool:
        return False

    def gen_clarify_question(self, *args: Any, **kwargs: Any) -> Optional[str]:  # pragma: no cover - not used
        return None


def _make_result(items: List[HAMDItem]) -> HAMDResult:
    return HAMDResult(
        items=items,
        total_score=HAMDTotal(
            得分序列="0", pre_correction_total=0, corrected_total=0, correction_basis=""
        ),
    )


def test_clarify_does_not_jump_to_previous_items() -> None:
    orchestrator = LangGraphMini.__new__(LangGraphMini)
    orchestrator.deepseek = _DummyDeepSeek()
    orchestrator.ITEM_NAMES = {3: "自杀倾向", 15: "疑病倾向"}

    state = SessionState(sid="sid", index=15)

    result = _make_result(
        [
            HAMDItem(
                item_id=3,
                symptom_summary="信息有限",
                dialogue_evidence="描述不足",
                evidence_refs=[],
                score=0,
                score_type="类型4",
                score_reason="缺少信息",
                clarify_need="频次",
            ),
            HAMDItem(
                item_id=15,
                symptom_summary="描述完整",
                dialogue_evidence="已经说明",
                evidence_refs=[],
                score=2,
                score_type="类型1",
                score_reason="完整",
                clarify_need=None,
            ),
        ]
    )

    clarify = LangGraphMini._clarify_from_analysis(orchestrator, state, result, [])

    assert clarify is None


def test_fallback_flow_preserves_existing_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = LangGraphMini.__new__(LangGraphMini)
    orchestrator.deepseek = _DummyDeepSeek()
    orchestrator.repo = _DummyRepo()

    captured: Dict[str, Any] = {}

    def _fake_make_response(
        self,
        sid: str,
        state: SessionState,
        text: str,
        *,
        turn_type: str = "ask",
        extra: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        payload = {
            "next_utterance": text,
            "analysis": state.analysis,
            "turn_type": turn_type,
        }
        if extra:
            payload.update(extra)
        captured.update(payload)
        return payload

    monkeypatch.setattr(LangGraphMini, "_make_response", _fake_make_response)

    state = SessionState(sid="sid", index=1)
    state.analysis = {"items": ["previous"]}

    monkeypatch.setattr(LangGraphMini, "_run_deepseek_analysis", lambda self, dialogue: None)
    monkeypatch.setattr(
        LangGraphMini,
        "_score_current_item",
        lambda self, state, transcripts, dialogue: None,
    )

    orchestrator._fallback_flow(
        sid="sid",
        state=state,
        item_id=1,
        scoring_segments=[],
        dialogue=[],
        transcripts=[],
        user_text="描述",
    )

    assert state.analysis == {"items": ["previous"]}
    assert captured["analysis"] == {"items": ["previous"]}
