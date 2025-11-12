from __future__ import annotations

import json
import os
from typing import Dict, Optional

from .langgraph_core.nodes_output import OutputNode
from .langgraph_core.nodes_risk import RiskNode
from .langgraph_core.nodes_score import ScoreNode
from .langgraph_core.nodes_strategy import StrategyNode
from .langgraph_core.patient_context import reinforce_with_context
from .langgraph_core.state_types import SessionState
from .langgraph_core.utils import now_iso, save_snapshot, write_jsonl

_BASE = os.path.dirname(__file__)
_CONFIG_DIR = os.path.join(_BASE, "config")
_LOG_PATH = os.path.join(_BASE, "logs", "session_log.jsonl")
_SNAPSHOT_DIR = os.path.join(_BASE, "state_snapshots")


class LangGraphCoordinator:
    """High-level orchestrator bridging the individual LangGraph nodes."""

    def __init__(self, total_items: int = 17, sid: Optional[str] = None, template_dir: Optional[str] = None) -> None:
        templates = template_dir or _CONFIG_DIR
        self.strategy = StrategyNode(templates)
        self.risk = RiskNode()
        self.score = ScoreNode()
        self.output = OutputNode()
        self.state = SessionState(sid=sid or now_iso(), total=total_items)

    # ------------------------------------------------------------------
    def step(self, role: str, text: Optional[str] = None) -> Dict[str, object]:
        if role not in {"agent", "user"}:
            raise ValueError(f"unsupported role: {role}")

        payload: Dict[str, object]
        if role == "user":
            self.state.last_role = "user"
            risk_payload = self.risk.check(self.state, text)
            if risk_payload:
                payload = risk_payload
            else:
                payload = self.strategy.run(self.state, text)
        else:
            payload = self.strategy.run(self.state, None)

        record = {
            "sid": self.state.sid,
            "role": role,
            "text": text,
            "payload": payload,
            "state": self.state.as_dict(),
        }
        write_jsonl(_LOG_PATH, record)
        snapshot_path = os.path.join(
            _SNAPSHOT_DIR, self.state.sid, f"turn_{self.state.index}_{role}.json"
        )
        save_snapshot(snapshot_path, record)
        return self.output.make_response(self.state, payload)

    # ------------------------------------------------------------------
    def next_item(self) -> Dict[str, object]:
        payload = self.strategy.finalize_item(self.state)
        current_index = self.state.index
        self.state.index += 1
        if self.state.index > self.state.total:
            self.state.completed = True
            self.score.parallel_score(self.state)
        else:
            self.state.patient_context = reinforce_with_context(
                self.state.patient_context, self.state.item_contexts
            )

        record = {
            "sid": self.state.sid,
            "event": "next_item",
            "item_index": current_index,
            "payload": payload,
            "state": self.state.as_dict(),
        }
        write_jsonl(_LOG_PATH, record)
        snapshot_path = os.path.join(
            _SNAPSHOT_DIR, self.state.sid, f"turn_{current_index}_next.json"
        )
        save_snapshot(snapshot_path, record)
        response_payload = {"event": "next_item", **payload}
        return self.output.make_response(self.state, response_payload)


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
