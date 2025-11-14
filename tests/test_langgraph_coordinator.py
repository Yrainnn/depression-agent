from __future__ import annotations

from pathlib import Path

import pytest

from services.orchestrator import langgraph_main as coordinator_module


@pytest.fixture()
def patched_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log_path = tmp_path / "log.jsonl"
    snap_dir = tmp_path / "snaps"
    monkeypatch.setattr(coordinator_module, "_LOG_PATH", str(log_path))
    monkeypatch.setattr(coordinator_module, "_SNAPSHOT_DIR", str(snap_dir))
    return log_path, snap_dir


def test_langgraph_basic_flow(patched_paths):
    coord = coordinator_module.LangGraphCoordinator(total_items=1, template_dir="services/orchestrator/config")
    first = coord.step(role="agent")
    assert first["waiting_for_user"] is True
    assert first["ask"]

    answer = "最近心情很低落，凌晨会醒来，我给自己打7分"
    second = coord.step(role="user", text=answer)
    assert "branch" in second
    assert second.get("clarify") in {True, False}

    if second.get("clarify"):
        assert second.get("ask")
        clarified = coord.step(role="user", text="是的，我就是觉得很难受")
        assert clarified.get("clarify") is False
        assert clarified.get("next_strategy")
        follow = coord.step(role="agent")
        assert follow.get("ask")
    else:
        follow = coord.step(role="agent")
        assert follow.get("ask")

    final = coord.next_item()
    assert final["event"] == "next_item"
    assert coord.state.completed is True
    analysis = coord.state.analysis
    assert analysis is not None
    assert analysis["total_score"]["items"][0]["item_id"] == 1
    assert analysis["total_score"]["items"][0]["score"] >= 0
    item_ctx = coord.state.item_contexts[1]
    assert item_ctx.patient_snapshot
    assert coord.state.patient_context.conversation_summary
    assert coord.state.patient_context.narrative_themes


def test_risk_gate_triggers(patched_paths):
    coord = coordinator_module.LangGraphCoordinator(total_items=1, template_dir="services/orchestrator/config")
    coord.step(role="agent")
    payload = coord.step(role="user", text="我想结束生命")
    # 风险节点当前为占位实现，直接进入策略流程
    assert "event" not in payload
    assert not coord.state.patient_context.active_risks
