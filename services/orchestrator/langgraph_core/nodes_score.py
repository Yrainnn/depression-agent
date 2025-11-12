from __future__ import annotations

from typing import Dict, List

from .llm_tools import LLM
from .state_types import SessionState


class ScoreNode:
    """Score completed items. Implementation is sequential but thread-safe."""

    def parallel_score(self, state: SessionState) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for item_id, context in state.item_contexts.items():
            dialogue_text = "\n".join(
                f"{turn.get('role')}: {turn.get('text', '')}"
                for turn in context.dialogue
                if isinstance(turn, dict)
            )
            payload = {
                "item_id": item_id,
                "item_name": context.item_name,
                "facts": context.facts,
                "themes": context.themes,
                "summary": context.summary,
                "dialogue": dialogue_text,
            }
            result = LLM.call("score_item", payload)
            if not isinstance(result, dict):
                result = {"item_id": item_id, "score": 0}
            result.setdefault("item_id", item_id)
            results.append(result)
        total = sum(int(entry.get("score", 0)) for entry in results)
        state.analysis = {"total_score": {"sum": total, "items": results}}
        return results
