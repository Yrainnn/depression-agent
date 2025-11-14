from __future__ import annotations

from typing import Any, Dict

from packages.common.config import settings
from services.tts.tts_adapter import TTSAdapter

from ..media import build_media_payload
from ..state_types import SessionState
from ..utils import now_iso
from .base_node import Node


class OutputNode(Node):
    """统一响应输出节点"""

    def __init__(self, name: str):
        super().__init__(name)
        self._tts_adapter = TTSAdapter()
        self._digital_human_enabled = bool(
            getattr(settings, "digital_human_enabled", False)
        )

    def run(self, state: SessionState, **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = kwargs.get("payload", {}) or {}
        response: Dict[str, Any] = {
            "ts": now_iso(),
            "sid": state.sid,
            "progress": {"index": state.index, "total": state.total},
            "waiting_for_user": state.waiting_for_user,
            "current_item": {"id": state.index, "name": state.current_item_name},
            "current_strategy": state.current_strategy,
            "patient_context": {
                "summary": state.patient_context.conversation_summary,
                "themes": list(state.patient_context.narrative_themes),
                "risks": list(state.patient_context.active_risks),
            },
        }
        response.update(payload)
        if state.analysis and "analysis" not in response:
            response["analysis"] = state.analysis
        final_message = response.get("final_message")
        if state.completed and not final_message:
            final_message = "评估结束,感谢您的参与."
            response["final_message"] = final_message
        if state.completed:
            if state.report_result:
                response.setdefault("report_generated", True)
                response.setdefault("report", state.report_result)
                url = state.report_result.get("report_url")
                if isinstance(url, str) and url:
                    response.setdefault("report_url", url)
            elif "report_generated" not in response:
                response["report_generated"] = False
        if response.get("final_message"):
            state.last_agent_text = response["final_message"]
            if "final_message_media" not in response:
                media = build_media_payload(
                    self._tts_adapter,
                    state.sid,
                    response["final_message"],
                    digital_human_enabled=self._digital_human_enabled,
                )
                if media:
                    response["final_message_media"] = media
            media_payload = response.get("final_message_media") or {}
            if isinstance(media_payload, dict):
                response.setdefault("final_message_media_type", media_payload.get("media_type"))
                if media_payload.get("tts_url"):
                    response.setdefault("final_message_tts_url", media_payload.get("tts_url"))
                if media_payload.get("video_url"):
                    response.setdefault("final_message_video_url", media_payload.get("video_url"))
        return response
