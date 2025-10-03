from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from services.llm.json_client import AnalysisResult, client as llm_client
from services.risk.engine import engine as risk_engine
from services.store.repository import repository

LOGGER = logging.getLogger(__name__)


@dataclass
class OrchestratorState:
    session_id: str
    stage: str = "collecting"
    clarifications_done: int = 0
    max_clarifications: int = 2
    completed: bool = False
    last_utt_index: int = 0
    pending_gap: Optional[str] = None
    summary: Optional[str] = None
    scores: List[Dict[str, object]] = field(default_factory=list)


class LangGraphMini:
    """A pragmatic orchestrator mimicking the LangGraph flow."""

    def __init__(self) -> None:
        self.repo = repository
        self.window_n = self._read_int_env("WINDOW_N", default=8)
        self.window_seconds = self._read_int_env("WINDOW_SECONDS", default=90)

        self.clarification_questions = {
            "frequency": "这种情况大概多久发生一次？",
            "duration": "这种状态持续了多长时间？",
            "severity": "这种感受对你的生活影响有多大？",
            "negation": "有没有出现好转或者缓解的时候？",
        }
        self.gap_keywords = {
            "frequency": ["每天", "经常", "偶尔"],
            "duration": ["两周", "一个月", "几天", "半年"],
            "severity": ["严重", "明显", "影响"],
            "negation": ["没有", "不会", "已经好"],
        }

    # ------------------------------------------------------------------
    def step(
        self,
        session_id: str,
        user_text: Optional[str] = None,
        segments: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        state = self._load_state(session_id)
        LOGGER.debug("Loaded state: %s", state)

        prepared_segments = self._prepare_segments(state, user_text, segments)
        for segment in prepared_segments:
            self.repo.append_transcript(session_id, segment)
            LOGGER.debug("Appended transcript: %s", segment)

        transcripts = self.repo.get_transcripts(session_id)
        scoring_segments = self._latest_segments(
            transcripts, self.window_n, self.window_seconds
        )

        for segment in prepared_segments:
            text = segment.get("text") or ""
            if not text:
                continue
            risk = risk_engine.evaluate(text)
            if risk.level == "high":
                self.repo.append_risk_event(
                    session_id,
                    {"level": risk.level, "triggers": risk.triggers},
                )
                state.stage = "risk_alert"
                state.completed = True
                self._persist_state(state)
                return {
                    "next_utterance": "我已经记录到紧急风险，请立即寻求当地紧急服务帮助。",
                    "progress": {"stage": state.stage, "completed": state.completed},
                    "risk": {"level": risk.level, "triggers": risk.triggers},
                }

        analysis = self._run_analysis(scoring_segments)
        if analysis:
            state.summary = analysis.summary
            state.scores = [score.dict() for score in analysis.scores]
            self.repo.save_scores(session_id, state.scores)

        next_gap = self._determine_gap(transcripts)
        response = self._advance_flow(state, next_gap, analysis)
        self._persist_state(state)
        return response

    # ------------------------------------------------------------------
    def _run_analysis(self, transcripts: List[Dict[str, object]]) -> Optional[AnalysisResult]:
        if not transcripts:
            return None
        try:
            return llm_client.analyze_transcript(transcripts)
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("LLM analysis failed: %s", exc)
            return None

    def _determine_gap(self, transcripts: List[Dict[str, object]]) -> Optional[str]:
        text = " ".join(seg.get("text", "") for seg in transcripts)
        missing: List[str] = []
        for gap, keywords in self.gap_keywords.items():
            if not any(keyword in text for keyword in keywords):
                missing.append(gap)
        return missing[0] if missing else None

    def _advance_flow(
        self,
        state: OrchestratorState,
        next_gap: Optional[str],
        analysis: Optional[AnalysisResult],
    ) -> Dict[str, object]:
        if state.completed:
            return {
                "next_utterance": state.summary or "感谢分享，我们的访谈已完成。",
                "progress": {"stage": state.stage, "completed": True},
            }

        if next_gap and state.clarifications_done < state.max_clarifications:
            question = self.clarification_questions[next_gap]
            state.clarifications_done += 1
            state.stage = "clarifying"
            state.pending_gap = next_gap
            return {
                "next_utterance": question,
                "progress": {
                    "stage": state.stage,
                    "completed": False,
                    "clarifications_done": state.clarifications_done,
                    "clarifications_remaining": state.max_clarifications - state.clarifications_done,
                },
            }

        state.stage = "summary"
        state.completed = True
        message = state.summary or "感谢分享，我们的访谈已完成。"
        payload: Dict[str, object] = {
            "next_utterance": message,
            "progress": {"stage": state.stage, "completed": state.completed},
        }
        if analysis is not None:
            payload["analysis"] = analysis.dict()
        return payload

    def _prepare_segments(
        self,
        state: OrchestratorState,
        user_text: Optional[str],
        segments: Optional[List[Dict[str, object]]],
    ) -> List[Dict[str, object]]:
        prepared: List[Dict[str, object]] = []
        incoming: List[Dict[str, object]] = []

        if segments:
            incoming.extend(segments)
        if user_text:
            incoming.append({"text": user_text, "speaker": "patient"})

        for segment in incoming:
            state.last_utt_index += 1
            normalized = dict(segment)
            normalized.setdefault("speaker", "patient")
            normalized["utt_id"] = f"u{state.last_utt_index}"
            prepared.append(normalized)

        return prepared

    def _load_state(self, session_id: str) -> OrchestratorState:
        raw = self.repo.load_session_state(session_id)
        state = OrchestratorState(session_id=session_id)
        if raw:
            state.stage = raw.get("stage", state.stage)
            state.clarifications_done = raw.get("clarifications_done", state.clarifications_done)
            state.max_clarifications = raw.get("max_clarifications", state.max_clarifications)
            state.completed = raw.get("completed", state.completed)
            state.last_utt_index = raw.get("last_utt_index", state.last_utt_index)
            state.pending_gap = raw.get("pending_gap", state.pending_gap)
            state.summary = raw.get("summary", state.summary)
            state.scores = raw.get("scores", state.scores)
        return state

    def _persist_state(self, state: OrchestratorState) -> None:
        self.repo.save_session_state(
            state.session_id,
            {
                "stage": state.stage,
                "clarifications_done": state.clarifications_done,
                "max_clarifications": state.max_clarifications,
                "completed": state.completed,
                "last_utt_index": state.last_utt_index,
                "pending_gap": state.pending_gap,
                "summary": state.summary,
                "scores": state.scores,
            },
        )

    def _latest_segments(
        self,
        transcripts: List[Dict[str, object]],
        max_items: int,
        max_seconds: int,
    ) -> List[Dict[str, object]]:
        if not transcripts:
            return []

        if max_items <= 0:
            max_items = len(transcripts)

        cutoff: Optional[float] = None
        if max_seconds > 0:
            last_end = self._segment_end(transcripts[-1])
            if last_end is not None:
                cutoff = last_end - float(max_seconds)

        window: List[Dict[str, object]] = []
        for segment in reversed(transcripts):
            if len(window) >= max_items:
                break

            end_time = self._segment_end(segment)
            if cutoff is not None and end_time is not None and end_time < cutoff:
                break

            window.append(segment)

        return list(reversed(window))

    @staticmethod
    def _segment_end(segment: Dict[str, object]) -> Optional[float]:
        ts = segment.get("ts")
        if isinstance(ts, (list, tuple)) and ts:
            for value in reversed(ts):
                if isinstance(value, (int, float)):
                    return float(value)
        return None

    @staticmethod
    def _read_int_env(name: str, *, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            return int(raw)
        except ValueError:
            LOGGER.warning("Invalid value for %s: %s; using default %s", name, raw, default)
            return default


orchestrator = LangGraphMini()
