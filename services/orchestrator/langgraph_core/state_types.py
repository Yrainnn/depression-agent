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

    def snapshot_for_item(self) -> Dict[str, Any]:
        """Return a shallow copy of the short-term state for long-term storage."""

        return {
            "summary": self.conversation_summary,
            "themes": list(self.narrative_themes),
            "facts": dict(self.structured_facts),
            "risks": list(self.active_risks),
        }

    def integrate_item_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Fold a long-term snapshot back into the short-term context."""

        summary = snapshot.get("summary")
        if isinstance(summary, str) and summary.strip():
            base = self.conversation_summary.strip()
            combined = f"{base} / {summary.strip()}" if base else summary.strip()
            self.conversation_summary = combined[-600:]

        themes = snapshot.get("themes")
        if isinstance(themes, list):
            for theme in themes:
                if isinstance(theme, str) and theme and theme not in self.narrative_themes:
                    self.narrative_themes.append(theme)

        facts = snapshot.get("facts")
        if isinstance(facts, dict):
            self.structured_facts.update(facts)

        risks = snapshot.get("risks")
        if isinstance(risks, list):
            for risk in risks:
                if isinstance(risk, str) and risk and risk not in self.active_risks:
                    self.active_risks.append(risk)

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
    risks: List[str] = field(default_factory=list)
    patient_snapshot: Dict[str, Any] = field(default_factory=dict)

    def absorb_patient_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Merge patient short-term context into this long-term record."""

        if not snapshot:
            return

        self.patient_snapshot = dict(snapshot)

        summary = snapshot.get("summary")
        if isinstance(summary, str) and summary.strip():
            self.summary = summary.strip()[-400:]

        facts = snapshot.get("facts")
        if isinstance(facts, dict):
            self.facts.update(facts)

        themes = snapshot.get("themes")
        if isinstance(themes, list):
            ordered = list(dict.fromkeys([t for t in themes if isinstance(t, str) and t]))
            if ordered:
                self.themes = ordered

        risks = snapshot.get("risks")
        if isinstance(risks, list):
            self.risks = list(dict.fromkeys([r for r in risks if isinstance(r, str) and r]))

    def export_for_patient(self) -> Dict[str, Any]:
        """Return a condensed snapshot for refreshing the patient context."""

        snapshot: Dict[str, Any] = {}
        if self.summary:
            snapshot["summary"] = self.summary
        elif self.patient_snapshot.get("summary"):
            snapshot["summary"] = self.patient_snapshot["summary"]

        if self.themes:
            snapshot["themes"] = list(self.themes)
        elif self.patient_snapshot.get("themes"):
            snapshot["themes"] = list(self.patient_snapshot["themes"])

        if self.facts:
            snapshot["facts"] = dict(self.facts)
        elif self.patient_snapshot.get("facts"):
            snapshot["facts"] = dict(self.patient_snapshot["facts"])

        if self.risks:
            snapshot["risks"] = list(self.risks)
        elif self.patient_snapshot.get("risks"):
            snapshot["risks"] = list(self.patient_snapshot["risks"])

        return snapshot


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
