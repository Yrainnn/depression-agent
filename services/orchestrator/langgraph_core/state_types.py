from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PatientContext:
    """Short-term memory used during question generation."""

    structured_facts: Dict[str, Any] = field(default_factory=dict)
    clarifications: List[str] = field(default_factory=list)
    narrative_themes: List[str] = field(default_factory=list)
    active_risks: List[str] = field(default_factory=list)
    conversation_summary: str = ""
    pending_clarifications: List[str] = field(default_factory=list)

    def to_prompt_snippet(self) -> str:
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
    """Long-term context, finalised when an item is completed."""

    item_id: int = 0
    item_name: str = ""
    dialogue: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""
    facts: Dict[str, Any] = field(default_factory=dict)
    themes: List[str] = field(default_factory=list)


@dataclass
class SessionState:
    """Top level orchestrator state shared across LangGraph nodes."""

    sid: str = ""
    index: int = 1
    total: int = 17

    current_item_name: str = ""
    current_strategy: str = ""
    strategy_substep_idx: int = 0
    strategy_history: List[str] = field(default_factory=list)

    clarify_count: int = 0
    waiting_for_user: bool = False
    completed: bool = False

    last_user_text: str = ""
    last_agent_text: str = ""
    last_role: str = "agent"

    analysis: Optional[Dict[str, Any]] = None

    patient_context: PatientContext = field(default_factory=PatientContext)
    item_contexts: Dict[int, ItemContext] = field(default_factory=dict)

    risk_recent_hits: int = 0

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["patient_context"] = asdict(self.patient_context)
        data["item_contexts"] = {k: asdict(v) for k, v in self.item_contexts.items()}
        return data
