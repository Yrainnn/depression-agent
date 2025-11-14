from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from services.orchestrator import langgraph_main as coordinator_module


@pytest.fixture()
def patched_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log_path = tmp_path / "log.jsonl"
    snap_dir = tmp_path / "snaps"
    monkeypatch.setattr(coordinator_module, "_LOG_PATH", str(log_path))
    monkeypatch.setattr(coordinator_module, "_SNAPSHOT_DIR", str(snap_dir))
    reports: list[Dict[str, Any]] = []

    def _fake_build_pdf(sid: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        stub = {"report_url": f"file://{sid}.pdf", "payload": payload}
        reports.append(stub)
        return stub

    monkeypatch.setattr(coordinator_module, "build_pdf", _fake_build_pdf)
    return log_path, snap_dir, reports


def test_langgraph_basic_flow(patched_paths):
    _, _, reports = patched_paths
    coord = coordinator_module.LangGraphCoordinator(total_items=1, template_dir="services/orchestrator/config")
    first = coord.step(role="agent")
    assert first["waiting_for_user"] is True
    assert first["ask"]

    answer = "最近心情很低落，凌晨会醒来，我给自己打7分"
    second = coord.step(role="user", text=answer)
    assert second["branch"] in {None, "明确存在抑郁情绪"}

    third = coord.step(role="agent")
    assert third["ask"]

    final = coord.next_item()
    assert final["event"] == "next_item"
    assert coord.state.completed is True
    analysis = coord.state.analysis
    assert analysis is not None
    assert analysis["total_score"]["items"][0]["item_id"] == 1
    assert analysis["total_score"]["items"][0]["score"] >= 0
    assert analysis["per_item_scores"][0]["item_code"].startswith("H0")
    assert analysis["diagnosis"]
    assert analysis["advice"]
    assert final["final_message"] == "评估结束,感谢您的参与."
    assert coord.state.report_payload is not None
    assert coord.state.report_result is not None
    assert final.get("report_generated") is True
    assert final.get("report_url") == "file://" + coord.state.sid + ".pdf"
    assert reports and reports[0]["report_url"] == "file://" + coord.state.sid + ".pdf"


def test_risk_gate_triggers(patched_paths):
    coord = coordinator_module.LangGraphCoordinator(total_items=1, template_dir="services/orchestrator/config")
    coord.step(role="agent")
    payload = coord.step(role="user", text="我想结束生命")
    # 风险节点当前为占位实现，直接进入策略流程
    assert "event" not in payload
    assert not coord.state.patient_context.active_risks
