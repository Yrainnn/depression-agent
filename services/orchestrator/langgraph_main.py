from __future__ import annotations

import json
import os
from typing import Dict, Optional

from .langgraph_core.context.item_context import finalize_item_context
from .langgraph_core.graphs.main_graph import GraphRuntime
from .langgraph_core.nodes.node_output import OutputNode
from .langgraph_core.nodes.node_score import ScoreNode
from .langgraph_core.state_types import SessionState
from .langgraph_core.utils import now_iso, save_snapshot, write_jsonl

_BASE = os.path.dirname(__file__)
_CONFIG_DIR = os.path.join(_BASE, "config")
_LOG_PATH = os.path.join(_BASE, "logs", "session_log.jsonl")
_SNAPSHOT_DIR = os.path.join(_BASE, "state_snapshots")


class LangGraphCoordinator:
    """Coordinator backed by a LangGraph StateGraph runtime."""

    def __init__(
        self,
        total_items: int = 17,
        sid: Optional[str] = None,
        template_dir: Optional[str] = None,
    ) -> None:
        self.state = SessionState(sid=sid or now_iso(), total=total_items)
        self.template_dir = template_dir or _CONFIG_DIR
        self.runtime = GraphRuntime(self.template_dir)
        self.score_node = ScoreNode("score_parallel")
        self.output_node = OutputNode("output")

    def step(self, role: str, text: Optional[str] = None) -> Dict[str, object]:
        payload = self.runtime.invoke(self.state, role=role, text=text)
        response = payload
        record = {
            "sid": self.state.sid,
            "role": role,
            "text": text,
            "payload": payload,
            "state": self.state.as_dict(),
        }
        write_jsonl(_LOG_PATH, record)
        snapshot_path = os.path.join(_SNAPSHOT_DIR, self.state.sid, f"turn_{self.state.index}_{role}.json")
        save_snapshot(snapshot_path, record)
        return response

    def next_item(self) -> Dict[str, object]:
        payload = finalize_item_context(self.state)
        current_index = self.state.index
        self.state.index += 1
        if self.state.index > self.state.total:
            self.state.completed = True
            self.score_node.run(self.state)
        else:
            self.state.current_strategy = ""
            self.state.current_template = None
            self.state.waiting_for_user = False
            self.state.strategy_sequence = []
            self.state.strategy_graph = {}
            self.state.strategy_map = {}
            self.state.default_next_strategy = ""
            self.state.strategy_prompt_overrides.clear()
            self.state.current_branches = []
            self.state.pending_strategy = ""
            self.state.clarify_attempts.clear()

        record = {
            "sid": self.state.sid,
            "event": "next_item",
            "item_index": current_index,
            "payload": payload,
            "state": self.state.as_dict(),
        }
        write_jsonl(_LOG_PATH, record)
        snapshot_path = os.path.join(_SNAPSHOT_DIR, self.state.sid, f"turn_{current_index}_next.json")
        save_snapshot(snapshot_path, record)
        response_payload = {"event": "next_item", **payload}
        response = self.output_node.run(self.state, payload=response_payload)
        return response


if __name__ == "__main__":
    coordinator = LangGraphCoordinator(total_items=1)
    first = coordinator.step(role="agent")
    print("Agent:", json.dumps(first, ensure_ascii=False))
    second = coordinator.step(role="user", text="最近心情很低落，尤其凌晨会醒来")
    print("User:", json.dumps(second, ensure_ascii=False))
    third = coordinator.next_item()
    print("Next:", json.dumps(third, ensure_ascii=False))
    if coordinator.state.completed:
        print("Analysis:", json.dumps(coordinator.state.analysis, ensure_ascii=False))
