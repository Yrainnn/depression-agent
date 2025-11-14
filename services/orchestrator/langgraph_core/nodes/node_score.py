from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from ..llm_tools import LLM, ScoreItemTool
from ..state_types import SessionState
from .base_node import Node


class ScoreNode(Node):
    """并行评分节点"""

    def __init__(self, name: str, max_workers: int = 4):
        super().__init__(name)
        # 默认至少启动 3 个并行线程，以满足批量评分需求
        self.max_workers = max(3, max_workers)

    def _score_single_item(self, item_id: int, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        result = LLM.call(ScoreItemTool, payload) or {}
        return item_id, result

    def run(self, state: SessionState, **_: Any) -> Dict[str, Any]:
        if not state.item_contexts:
            state.analysis = {"total_score": {"sum": 0, "items": []}}
            return {"analysis": state.analysis}

        ordered_contexts = sorted(state.item_contexts.items())
        payloads: List[Tuple[int, Dict[str, Any]]] = []
        for item_id, ctx in ordered_contexts:
            payloads.append(
                (
                    item_id,
                    {
                        "item_name": ctx.item_name,
                        "facts": ctx.facts,
                        "themes": ctx.themes,
                        "summary": ctx.summary,
                        "dialogue": ctx.dialogue,
                        "risks": ctx.risks,
                    },
                )
            )

        worker_count = min(self.max_workers, len(payloads)) or 1

        scored_items: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(self._score_single_item, item_id, payload): item_id for item_id, payload in payloads}
            for future in as_completed(futures):
                item_id = futures[future]
                try:
                    _, result = future.result()
                except Exception:
                    result = {}
                scored_items[item_id] = result

        items: List[Dict[str, Any]] = []
        for item_id, _ in ordered_contexts:
            result = scored_items.get(item_id, {})
            score = result.get("score", 0)
            items.append({"item_id": item_id, "score": score, "raw": result})

        state.analysis = {"total_score": {"sum": sum(item["score"] for item in items), "items": items}}
        return {"analysis": state.analysis}
