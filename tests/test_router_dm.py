from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import pytest

from apps.api import router_dm


class _DummyPatientContext:
    def snapshot_for_item(self) -> Dict[str, Any]:
        return {"summary": "", "themes": [], "facts": {}, "risks": []}


class _StubState:
    def __init__(self) -> None:
        self.sid = "session-test"
        self.index = 1
        self.total = 17
        self.current_item_name = "测试条目"
        self.waiting_for_user = False
        self.completed = False
        self.patient_context = _DummyPatientContext()


class _StubCoordinator:
    def __init__(self) -> None:
        self.state = _StubState()
        self.calls: List[Tuple[str, Optional[str]]] = []

    def step(self, role: str, text: Optional[str] = None) -> Dict[str, Any]:
        self.calls.append((role, text))
        base = {
            "ts": "2024-01-01T00:00:00Z",
            "sid": self.state.sid,
            "progress": {"index": self.state.index, "total": self.state.total},
            "current_item": {
                "id": self.state.index,
                "name": self.state.current_item_name,
            },
            "patient_context": self.state.patient_context.snapshot_for_item(),
        }

        if role == "user":
            self.state.waiting_for_user = False
            base.update(
                {
                    "waiting_for_user": False,
                    "risk_level": "high",
                    "risk_result": {
                        "risk_level": "high",
                        "message": "请立即寻求帮助",
                    },
                }
            )
            return base

        self.state.waiting_for_user = True
        base.update(
            {
                "waiting_for_user": True,
                "ask": "请继续描述您的睡眠情况。",
                "media": {
                    "media_type": "audio",
                    "tts_url": "file://ask.wav",
                },
            }
        )
        return base


@pytest.fixture()
def stub_coordinator(monkeypatch: pytest.MonkeyPatch) -> _StubCoordinator:
    coordinator = _StubCoordinator()

    async def _fake_get_coordinator(sid: str) -> _StubCoordinator:  # type: ignore[override]
        return coordinator

    monkeypatch.setattr(router_dm, "_get_coordinator", _fake_get_coordinator)
    # ensure each test starts with a clean coordinator registry lock/entries
    router_dm._COORDINATORS.clear()
    router_dm._SESSION_LOCKS.clear()

    return coordinator


def test_dm_step_initial_agent_turn(stub_coordinator: _StubCoordinator):
    payload = router_dm.DMStepPayload(sid="session-test")
    result = asyncio.run(router_dm.dm_step(payload))
    assert getattr(result, "ask") == "请继续描述您的睡眠情况。"
    assert getattr(result, "next_utterance") == "请继续描述您的睡眠情况。"
    assert getattr(result, "risk_flag") is False
    assert stub_coordinator.calls == [("agent", None)]


def test_dm_step_user_turn_merges_risk_and_question(stub_coordinator: _StubCoordinator):
    payload = router_dm.DMStepPayload(sid="session-test", text="最近总是半夜醒来")
    result = asyncio.run(router_dm.dm_step(payload))
    assert getattr(result, "ask") == "请继续描述您的睡眠情况。"
    assert getattr(result, "risk_level") == "high"
    assert getattr(result, "risk_flag") is True
    assert getattr(result, "next_utterance") == "请继续描述您的睡眠情况。"
    assert stub_coordinator.calls == [("user", "最近总是半夜醒来"), ("agent", None)]
