from __future__ import annotations

from typing import Any, Dict, Optional

from .state_types import SessionState
from .utils import now_iso


class OutputNode:
    """Build public responses from the current session state."""

    def make_response(
        self, state: SessionState, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "ts": now_iso(),
            "sid": state.sid,
            "progress": {"index": state.index, "total": state.total},
            "waiting_for_user": state.waiting_for_user,
            "current_item": {"id": state.index, "name": state.current_item_name},
            "current_strategy": state.current_strategy,
            "patient_context": {
                "summary": state.patient_context.conversation_summary,
                "themes": state.patient_context.narrative_themes,
                "risks": state.patient_context.active_risks,
            },
        }
        if payload:
            response.update(payload)
        if state.analysis:
            response["analysis"] = state.analysis
        return response
