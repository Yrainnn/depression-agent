from services.orchestrator.langgraph_core.reporting import prepare_report_payload
from services.orchestrator.langgraph_core.state_types import ItemContext, SessionState


def test_prepare_report_payload_collects_scores_and_reasons():
    state = SessionState(sid="demo")
    item_context = ItemContext(item_id=1, item_name="抑郁情绪")
    state.item_contexts[1] = item_context
    state.patient_context.conversation_summary = "患者情绪持续低落"
    state.analysis = {
        "total_score": {
            "sum": 3,
            "items": [
                {"item_id": 1, "score": 3, "raw": {"reason": "情绪明显低落"}},
            ],
        }
    }

    payload = prepare_report_payload(state)

    assert payload is not None
    assert payload["total_score"] == 3
    assert payload["items"][0]["item_id"] == 1
    assert payload["items"][0]["question"] == "抑郁情绪"
    assert payload["summary"] == "患者情绪持续低落"
    assert "情绪明显低落" in payload["opinion"]["rationale"]
