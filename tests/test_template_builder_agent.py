from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from services.orchestrator.langgraph_core import template_builder_agent as tba


@pytest.fixture(autouse=True)
def restore_llm(monkeypatch: pytest.MonkeyPatch):
    original_call = tba.LLM.call
    yield
    monkeypatch.setattr(tba.LLM, "call", original_call)


def test_template_builder_generates_yaml_and_descriptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_root = tmp_path / "templates"
    descriptions_path = tmp_path / "strategy_descriptions.py"
    agent = tba.TemplateBuilderAgent(str(output_root), descriptions_path=str(descriptions_path))

    llm_payload = {
        "yaml": "strategies:\n  S1:\n    prompts:\n      - question: \"测试\"\n",
        "strategy_descriptions": {
            "S1": {
                "name": "测试策略",
                "description": "用于评估测试场景的策略流程",
                "tone": "支持性、耐心",
            }
        },
    }

    def fake_call(tool, payload: dict) -> dict:
        assert tool is tba.TemplateBuilderTool
        assert "策略描述" in payload["prompt"]
        return {"text": json.dumps(llm_payload, ensure_ascii=False)}

    monkeypatch.setattr(tba.LLM, "call", fake_call)

    yaml_path = agent.build_from_text("策略草案", 7, project_name="测试项目")
    assert Path(yaml_path).exists()

    with open(yaml_path, "r", encoding="utf-8") as fh:
        yaml_content = fh.read()
    if tba.yaml:
        data = tba.yaml.safe_load(yaml_content)
    else:
        data = tba._safe_load_fallback(yaml_content)

    assert data["project_id"] == 7
    assert data["project_name"] == "测试项目"
    assert "S1" in (data.get("strategies") or {})

    descriptions_module = runpy.run_path(str(descriptions_path))
    descriptions = descriptions_module["STRATEGY_DESCRIPTIONS"]
    assert descriptions["S1"]["name"] == "测试策略"
    assert "tone" in descriptions["S1"]
