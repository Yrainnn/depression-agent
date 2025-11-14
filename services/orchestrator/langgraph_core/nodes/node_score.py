from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from ..llm_tools import LLM, ScoreItemTool
from ..state_types import SessionState
from ...questions_hamd17 import MAX_SCORE
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
            state.analysis = {
                "total_score": {"sum": 0, "max": sum(MAX_SCORE.values())},
                "per_item_scores": [],
            }
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
                        "dialogue": json.dumps(ctx.dialogue, ensure_ascii=False),
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

        total = 0.0
        items: List[Dict[str, Any]] = []
        rationale_lines: List[str] = []

        for item_id, ctx in ordered_contexts:
            result = scored_items.get(item_id, {}) or {}
            score_raw = result.get("score")
            try:
                score_value = float(score_raw) if score_raw is not None else None
            except (TypeError, ValueError):
                score_value = None

            if isinstance(score_value, (int, float)):
                total += float(score_value)

            reason = result.get("reason")
            reason_text = reason.strip() if isinstance(reason, str) else ""
            if reason_text:
                rationale_lines.append(f"H{item_id:02d}: {reason_text}")

            item_entry: Dict[str, Any] = {
                "item_id": item_id,
                "item_code": f"H{item_id:02d}",
                "question": ctx.item_name or f"条目 {item_id}",
                "score": score_value,
                "max_score": MAX_SCORE.get(item_id),
            }
            if reason_text:
                item_entry["reason"] = reason_text

            items.append(item_entry)

        total_score = round(total, 2)
        max_total = sum(MAX_SCORE.values())

        diagnosis: str
        advice: str
        if total_score <= 7:
            diagnosis = "无抑郁症状"
            advice = "情绪状态良好，无明显抑郁表现。建议保持规律作息与积极生活方式。"
        elif total_score <= 16:
            diagnosis = "轻度抑郁"
            advice = "出现轻度情绪低落，建议适度休息、增加社交活动，并考虑心理疏导。"
        elif total_score <= 23:
            diagnosis = "中度抑郁"
            advice = "表现出明显抑郁特征，建议及时寻求心理咨询或医生指导。"
        else:
            diagnosis = "重度抑郁"
            advice = "存在严重抑郁表现，建议尽快就医，进行专业评估与治疗。"

        items_for_analysis = [dict(entry) for entry in items]

        analysis: Dict[str, Any] = {
            "total_score": {"sum": total_score, "max": max_total, "items": items_for_analysis},
            "per_item_scores": items_for_analysis,
            "diagnosis": diagnosis,
            "advice": advice,
        }

        if rationale_lines:
            analysis["rationale"] = "\n".join(rationale_lines)

        state.analysis = analysis
        return {"analysis": state.analysis}
