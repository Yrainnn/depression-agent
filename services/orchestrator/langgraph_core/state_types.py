from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PatientContext:
    """短期记忆池：用于提问生成"""

    structured_facts: Dict[str, Any] = field(default_factory=dict)
    narrative_themes: List[str] = field(default_factory=list)
    conversation_summary: str = ""
    active_risks: List[str] = field(default_factory=list)

    def to_prompt_snippet(self) -> str:
        """Return a concise prompt fragment for LLM question generation."""
        parts: List[str] = []
        if self.conversation_summary:
            parts.append(f"摘要: {self.conversation_summary}")
        if self.narrative_themes:
            parts.append("主题: " + ", ".join(self.narrative_themes))
        if self.active_risks:
            parts.append("风险: " + ", ".join(self.active_risks))
        return "\n".join(parts)


@dataclass
class ItemContext:
    """长期记忆池：用于评分与报告"""

    item_id: int = 0
    item_name: str = ""
    dialogue: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""
    facts: Dict[str, Any] = field(default_factory=dict)
    themes: List[str] = field(default_factory=list)


@dataclass
class SessionState:
    sid: str
    index: int = 1
    total: int = 17
    current_item_name: str = ""
    current_strategy: str = ""
    waiting_for_user: bool = False
    completed: bool = False
    patient_context: PatientContext = field(default_factory=PatientContext)
    item_contexts: Dict[int, ItemContext] = field(default_factory=dict)
    current_template: Optional[Dict[str, Any]] = None
    current_branches: List[Dict[str, Any]] = field(default_factory=list)
    analysis: Optional[Dict[str, Any]] = None
    last_agent_text: str = ""
    last_user_text: str = ""
    branch_history: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["patient_context"] = asdict(self.patient_context)
        data["item_contexts"] = {k: asdict(v) for k, v in self.item_contexts.items()}
        return data
