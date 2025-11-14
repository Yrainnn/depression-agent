from __future__ import annotations

from typing import Dict, Any

from ..context.item_context import ensure_item_context
from ..state_types import SessionState
from ..utils import load_template
from .base_node import Node


class InitNode(Node):
    """加载模板、初始化会话指针"""

    def __init__(self, name: str, template_root: str):
        super().__init__(name)
        self.template_root = template_root

    def run(self, state: SessionState, **_: Any) -> Dict[str, Any]:
        template = load_template(self.template_root, state.index)
        state.current_template = template
        state.current_item_name = template.get("project_name", f"item_{state.index}")
        ensure_item_context(state)
        if not state.current_strategy:
            state.current_strategy = "S2"
        return {"item_name": state.current_item_name}
