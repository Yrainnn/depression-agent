from __future__ import annotations

from typing import Optional

from .state_types import SessionState


class RiskNode:
    """
    预留风险检测与干预节点。
    当前版本仅保留接口和调用点，不进行任何动作。
    """

    def __init__(self) -> None:
        # 未来可接入 LLM 风险评估 / Rule-based 规则 / 评分融合
        self.enabled = True

    def check(self, state: SessionState, user_text: Optional[str]) -> Optional[dict]:
        """
        占位实现：
        - 返回 None 表示当前不触发风险分支
        - 未来可以返回 {"type": "risk", "level": "...", "message": "..."}
        """
        return None

    def handle(self, state: SessionState, risk_payload: dict) -> dict:
        """
        风险事件处理子图（占位）
        - 未来用于异步告警、人工审查、记录、报告整合。
        """
        return {"status": "risk_subgraph_placeholder"}
