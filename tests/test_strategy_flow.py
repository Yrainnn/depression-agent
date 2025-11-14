from __future__ import annotations

from unittest.mock import patch
from services.orchestrator.langgraph_core.nodes.node_clarify import ClarifyNode
from services.orchestrator.langgraph_core.nodes.node_strategy import StrategyNode
from services.orchestrator.langgraph_core.nodes.node_update import UpdateNode
from services.orchestrator.langgraph_core.state_types import SessionState


def _build_state(template: dict) -> SessionState:
    state = SessionState(sid="test", index=1)
    state.current_template = template
    return state


def test_strategy_node_builds_graph_from_template():
    template = {
        "strategies": [
            {
                "id": "S2",
                "template": "最近两周，大部分时间您的心情是不是都很低落、忧郁？",
                "branches": [
                    {"condition": "明确存在抑郁情绪", "next": "S4"},
                    {
                        "condition": "否定或含糊",
                        "next": "S10",
                        "next_prompt": "情绪没受影响，那有没有觉得每天过得很空洞、乏味？",
                    },
                ],
            },
            {"id": "S4", "template": "您觉得这种难过和以往相比有什么不同？"},
            {"id": "S10", "template": "那有没有觉得每天过得很空洞、乏味？"},
        ]
    }
    state = _build_state(template)
    strategy_node = StrategyNode("strategy")

    payload = strategy_node.run(state, role="agent")

    assert state.current_strategy == "S2"
    assert state.strategy_sequence == ["S2", "S4", "S10"]
    assert payload["strategy_graph"]["S2"] == []
    assert payload["ask"]
    assert state.waiting_for_user is True


def test_branch_transition_applies_prompt_override():
    template = {
        "strategies": [
            {
                "id": "S2",
                "template": "最近两周，大部分时间您的心情是不是都很低落、忧郁？",
                "branches": [
                    {"condition": "明确存在抑郁情绪", "next": "S4"},
                    {
                        "condition": "否定或含糊",
                        "next": "S10",
                        "next_prompt": "情绪没受影响，那有没有觉得每天过得很空洞、乏味？",
                    },
                ],
            },
            {"id": "S4", "template": "您觉得这种难过和以往相比有什么不同？"},
            {"id": "S10", "template": "那有没有觉得每天过得很空洞、乏味？"},
        ]
    }
    state = _build_state(template)
    strategy_node = StrategyNode("strategy")
    clarify_node = ClarifyNode("clarify")
    update_node = UpdateNode("update")

    strategy_node.run(state, role="agent")

    answer = "情绪没受影响，没有觉得难过"
    clarify_result = clarify_node.run(state, user_text=answer)

    assert clarify_result["next_strategy"] == "S10"
    assert state.strategy_graph["S2"] == [{"to": "S10", "condition": "否定或含糊", "prompt_override": "情绪没受影响，那有没有觉得每天过得很空洞、乏味？"}]
    assert state.strategy_prompt_overrides["S10"].startswith("情绪没受影响")

    update_node.run(state, user_text=answer, **clarify_result)

    follow_up = strategy_node.run(state, role="agent")
    assert follow_up["strategy"] == "S10"
    assert "空洞" in (follow_up["ask"] or "")
    assert "S10" not in state.strategy_prompt_overrides


def test_default_next_moves_to_end():
    template = {
        "strategies": [
            {"id": "S1", "template": "我们聊到这里可以吗？", "next": "END"},
        ]
    }
    state = _build_state(template)
    strategy_node = StrategyNode("strategy")
    clarify_node = ClarifyNode("clarify")
    update_node = UpdateNode("update")

    strategy_node.run(state, role="agent")
    assert state.default_next_strategy == "END"

    clarify_result = clarify_node.run(state, user_text="好的")
    assert clarify_result["next_strategy"] == "END"
    assert state.strategy_graph["S1"] == [{"to": "END", "condition": "default_next"}]

    update_node.run(state, user_text="好的", **clarify_result)
    assert state.completed is True

    follow_up = strategy_node.run(state, role="agent")
    assert follow_up["ask"] is None
    assert follow_up["strategy"] == "END"


def test_clarify_retries_then_falls_back_to_default():
    template = {
        "strategies": [
            {
                "id": "S1",
                "template": "最近感觉怎么样？",
                "clarify_prompt": "能再具体说明一下吗？",
                "next": "S2",
                "branches": [{"condition": "肯定", "next": "S3"}],
            },
            {"id": "S2", "template": "谢谢分享，我们继续。"},
            {"id": "S3", "template": "很高兴听到这一点。"},
        ]
    }
    state = _build_state(template)
    state.max_clarify_attempts = 2
    strategy_node = StrategyNode("strategy")
    clarify_node = ClarifyNode("clarify")

    clarify_responses = iter(
        [
            {"matched": None, "next": None, "reason": "low_confidence"},
            {"matched": None, "next": None, "reason": "still_unclear"},
        ]
    )

    def fake_call(func: str, payload: dict) -> dict:
        if func == "generate":
            template_text = payload.get("template", "")
            return {"text": template_text}
        if func == "clarify_branch":
            return next(clarify_responses)
        if func == "match_condition":
            condition = payload.get("condition", "")
            answer = payload.get("answer", "")
            if "肯定" in condition:
                return {"match": "肯定" in answer or "有" in answer}
            return {"match": False}
        return {}

    strategy_path = "services.orchestrator.langgraph_core.nodes.node_strategy.LLM.call"
    clarify_path = "services.orchestrator.langgraph_core.nodes.node_clarify.LLM.call"

    with patch(strategy_path, side_effect=fake_call), patch(clarify_path, side_effect=fake_call):
        first_question = strategy_node.run(state, role="agent")
        assert first_question["ask"] == "最近感觉怎么样？"
        assert state.default_next_strategy == "S2"

        clarify_first = clarify_node.run(state, user_text="说不清楚")
        assert clarify_first["next_strategy"] is None
        assert clarify_first["clarify_prompt"] == "能再具体说明一下吗？"
        assert state.strategy_prompt_overrides["S1"] == "能再具体说明一下吗？"

        follow_up = strategy_node.run(state, role="agent")
        assert follow_up["ask"] == "能再具体说明一下吗？"
        assert "S1" not in state.strategy_prompt_overrides

        clarify_second = clarify_node.run(state, user_text="还是说不清")
        assert clarify_second["next_strategy"] == "S2"
        assert clarify_second["clarify_reason"] == "clarify_limit"
        assert state.current_strategy == "S2"
        assert state.strategy_graph["S1"] == [{"to": "S2", "condition": "clarify_limit"}]
        assert "S1" not in state.clarify_attempts
