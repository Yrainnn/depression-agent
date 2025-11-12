from __future__ import annotations

from typing import Dict, List

from .llm_tools import LLM
from .state_types import SessionState


class ScoreNode:
    """Score completed items. Implementation is sequential but thread-safe."""

    def parallel_score(self, state: SessionState) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        for item_id, context in state.item_contexts.items():
            items.append(
                LLM.score_item(
                    {
                        "item_id": item_id,
                        "facts": context.facts,
                        "themes": context.themes,
                        "summary": context.summary,
                    }
                )
            )
        total = sum(int(entry.get("score", 0)) for entry in items)
        state.analysis = {"total_score": {"sum": total, "items": items}}
        return items
