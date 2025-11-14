from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _reload_llm_module(monkeypatch: pytest.MonkeyPatch):
    module = importlib.import_module("services.orchestrator.langgraph_core.llm_tools")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "LLM", None, raising=False)
    return module


class _StubDeepSeek:
    def __init__(self) -> None:
        self.plan_turn_calls: List[Dict[str, Any]] = []

    def usable(self) -> bool:
        return True

    def plan_turn(
        self,
        *,
        dialogue: List[dict] | None = None,
        progress: Dict[str, Any] | None = None,
        prompt: str,
        **_: Any,
    ) -> Dict[str, Any]:
        self.plan_turn_calls.append(
            {
                "dialogue": dialogue or [],
                "progress": progress or {},
                "prompt": prompt,
            }
        )
        return {"question": "请问现在感觉如何？"}


def test_toolbox_fallback_branch(monkeypatch: pytest.MonkeyPatch):
    module = _reload_llm_module(monkeypatch)
    monkeypatch.setattr(module, "_deepseek_client", None, raising=False)

    toolbox = module.LLMToolBox()
    question = toolbox.call("generate", {"template": "请描述一下最近的情绪"})
    assert question["text"].startswith("请描述一下最近的情绪")
    assert question["text"].endswith("？")

    branches = [
        {"condition": "否定或含糊", "next": "S10"},
        {"condition": "明确存在抑郁情绪", "next": "S4"},
    ]
    clarified = toolbox.call("clarify_branch", {"answer": "没有太大影响", "branches": branches})
    assert clarified["next"] == "S10"
    assert clarified.get("clarify") is False
    assert "fallback" in clarified["reason"]

    clarify_question = toolbox.call(
        "clarify_question",
        {"context": "测试摘要", "template": "请描述感受", "answer": "说不清"},
    )
    assert "question" in clarify_question


def test_toolbox_deepseek_backend(monkeypatch: pytest.MonkeyPatch):
    module = _reload_llm_module(monkeypatch)
    responses = [
        {"risk_level": "none"},
        {"facts": {"self_rating": 3}},
        {"themes": ["绝望"]},
        {"summary": "旧摘要与新内容合并后的结果应被截断"},
        {"score": 3, "reason": "模型判断"},
        {
            "matched": "否定或含糊",
            "next": None,
            "clarify": True,
            "clarify_question": "能再详细说说吗？",
            "reason": "LLM 判断不明确",
        },
        {"question": "是否可以具体说明影响？"},
    ]
    chat_calls: List[Dict[str, Any]] = []

    def fake_chat(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        if not responses:
            raise AssertionError("缺少预置的 LLM 响应")
        kwargs.setdefault("max_tokens", 512)
        chat_calls.append({"prompt": prompt, **kwargs})
        return responses.pop(0)

    stub = _StubDeepSeek()
    monkeypatch.setattr(module, "_deepseek_client", stub, raising=False)
    monkeypatch.setattr(module._DeepSeekBackend, "_chat_json", fake_chat, raising=False)

    toolbox = module.LLMToolBox()
    generated = toolbox.call(
        "generate",
        {
            "context": "示例上下文",
            "template": "请简单描述一下今日心情",
            "dialogue": [{"role": "user", "content": "test"}],
            "progress": {"index": 1, "total": 17},
        },
    )
    assert generated["text"] == "请问现在感觉如何？"
    assert stub.plan_turn_calls
    assert "示例上下文" in stub.plan_turn_calls[0]["prompt"]

    risk = toolbox.call("risk_detect", {"text": "没有危险信号"})
    assert risk == {"risk_level": "none"}

    facts = toolbox.call("extract_facts", {"text": "我给自己打3分"})
    assert facts["facts"]["self_rating"] == 3

    themes = toolbox.call("identify_themes", {"text": "感觉很绝望"})
    assert themes["themes"] == ["绝望"]

    summary = toolbox.call(
        "summarize_context",
        {"prev": "旧摘要", "new": "新的内容", "limit": 10},
    )
    assert len(summary["summary"]) <= 10

    score = toolbox.call(
        "score_item",
        {
            "item_name": "项目1",
            "facts": {"a": 1},
            "themes": ["绝望"],
            "summary": "概要",
            "dialogue": "对话内容",
        },
    )
    assert score == {"score": 3, "reason": "模型判断"}
    assert chat_calls[-1]["max_tokens"] == 256

    clarified = toolbox.call(
        "clarify_branch",
        {
            "answer": "患者否认严重症状",
            "branches": [
                {"condition": "明确存在抑郁情绪", "next": "S4"},
                {"condition": "否定或含糊", "next": "S10"},
            ],
        },
    )
    assert clarified["matched"] == "否定或含糊"
    assert clarified["clarify"] is True
    assert clarified["clarify_question"] == "能再详细说说吗？"

    follow_up = toolbox.call(
        "clarify_question",
        {"context": "示例上下文", "template": "请描述影响", "answer": "不太确定"},
    )
    assert follow_up["question"] == "是否可以具体说明影响？"

    # 前七次调用依次对应 risk/facts/themes/summary/score/clarify/clarify_question
    assert len(chat_calls) == 7
    assert chat_calls[3]["max_tokens"] == 512
