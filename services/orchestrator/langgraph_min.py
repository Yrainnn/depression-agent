from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from packages.common.config import settings
from services.audio.asr_adapter import AsrError, StubASR
from services.risk.engine import engine as risk_engine
from services.store.repository import repository

LOGGER = logging.getLogger(__name__)


PHQ9: List[Tuple[str, str]] = [
    ("P1", "过去两周，做事时缺乏兴趣或乐趣？"),
    ("P2", "过去两周，感到情绪低落、沮丧或绝望？"),
    ("P3", "过去两周，入睡困难、易醒或睡眠过多？"),
    ("P4", "过去两周，感到疲倦或精力不足？"),
    ("P5", "过去两周，食欲不振或吃太多？"),
    ("P6", "过去两周，觉得自己很糟、失败或让家人失望？"),
    ("P7", "过去两周，注意力不集中，比如看电视或读书时？"),
    ("P8", "过去两周，行动或说话缓慢，或烦躁坐立不安？"),
    ("P9", "过去两周，有过伤害自己的想法？"),
]


CLARIFY_PROMPTS: Dict[str, str] = {
    "P1": "频率是几天、过半天、几乎每天？",
    "P2": "大概每周几天出现低落情绪？",
    "P3": "更常见是入睡难、早醒还是多睡？",
    "P4": "疲倦大概每周几天出现？",
    "P5": "更倾向食欲减退还是增多？",
    "P6": "此类感受每周大约几天？",
    "P7": "注意力不集中每周大致几天？",
    "P8": "更像变慢还是烦躁？每周几天？",
    "P9": "是否曾出现具体计划或准备？",
}

DEFAULT_CLARIFY = "能具体点吗？比如每周几天？"

VAGUE_PHRASES = [
    "还好",
    "一般",
    "差不多",
    "说不清",
    "不好说",
    "看情况",
    "可能吧",
    "偶尔吧",
    "有点吧",
    "还行",
    "凑合",
]

MISSING_INPUT_PROMPT = "未获取音频/文本，请重新描述一次好吗？"


SCORE_KEYWORDS: Dict[int, List[str]] = {
    3: ["几乎每天", "每天", "总是", "一直"],
    2: ["过半天", "经常", "很多天", "大部分时间"],
    1: ["几天", "有时", "偶尔", "有点"],
}

NEGATION_KEYWORDS = ["没有", "不太", "不算", "很少", "几乎没有", "没"]


@dataclass
class SessionState:
    sid: str
    index: int = 1
    total: int = len(PHQ9)
    clarify: int = 0
    scores_acc: List[Dict[str, object]] = field(default_factory=list)
    completed: bool = False
    last_utt_index: int = 0
    opinion: Optional[str] = None


class LangGraphMini:
    """A LangGraph-inspired orchestrator implementing the PHQ-9 flow."""

    def __init__(self) -> None:
        self.repo = repository
        self.window_n = self._read_int_env("WINDOW_N", default=8)
        self.window_seconds = self._read_int_env("WINDOW_SECONDS", default=90)
        self.stub_asr = StubASR()
        provider_name = getattr(settings, "ASR_PROVIDER", "")
        if provider_name and provider_name.lower() == "tingwu":
            try:
                from services.audio.asr_adapter import SDKTingWuASR

                self.asr = SDKTingWuASR(settings)
            except Exception as exc:  # pragma: no cover - configuration guard
                LOGGER.warning(
                    "Failed to initialise SDKTingWuASR, using stub instead: %s", exc
                )
                self.asr = self.stub_asr
        else:
            self.asr = self.stub_asr

    # ------------------------------------------------------------------
    def ask(self, sid: str) -> Dict[str, object]:
        state = self._load_state(sid)
        if state.completed:
            return self._complete_payload(state, "评估已完成，可导出报告。")
        item_id, question = self._current_item(state)
        LOGGER.debug("Ask called for %s at item %s", sid, item_id)
        self._persist_state(state)
        return {
            "next_utterance": question,
            "progress": {"index": state.index, "total": state.total},
            "risk_flag": False,
            "tts_text": None,
        }

    def step(
        self,
        sid: str,
        role: Optional[str] = None,
        text: Optional[str] = None,
        audio_ref: Optional[str] = None,
        segments: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        state = self._load_state(sid)
        LOGGER.debug("Loaded state for %s: %s", sid, state)

        if state.completed:
            return self._complete_payload(state, "评估已完成，可导出报告。")

        raw_segments: List[Dict[str, object]] = []
        if segments is not None:
            raw_segments.extend(segments)
        elif text or audio_ref:
            text_segments: List[Dict[str, object]] = []
            if text:
                text_segments = self.stub_asr.transcribe(text=text)

            audio_segments: List[Dict[str, object]] = []
            if audio_ref:
                try:
                    audio_segments = self.asr.transcribe(text=None, audio_ref=audio_ref)
                except AsrError as exc:
                    LOGGER.warning("ASR audio transcription failed for %s: %s", sid, exc)
                    if text_segments:
                        raw_segments.extend(text_segments)
                        text_segments = []
                    else:
                        self._persist_state(state)
                        return {
                            "next_utterance": MISSING_INPUT_PROMPT,
                            "progress": {"index": state.index, "total": state.total},
                            "risk_flag": False,
                            "tts_text": None,
                        }
                else:
                    raw_segments.extend(text_segments)
                    raw_segments.extend(audio_segments)
                    text_segments = []

            if text_segments:
                raw_segments.extend(text_segments)

        prepared_segments = self._prepare_segments(state, raw_segments)
        if prepared_segments:
            for segment in prepared_segments:
                self.repo.append_transcript(sid, segment)
                LOGGER.debug("Appended transcript for %s: %s", sid, segment)

        transcripts = self.repo.get_transcripts(sid)
        scoring_segments = self._latest_segments(
            transcripts, self.window_n, self.window_seconds
        )

        risk_payload = self._check_risk(sid, state, prepared_segments)
        if risk_payload is not None:
            self._persist_state(state)
            return risk_payload

        user_text = self._extract_user_text(prepared_segments)
        if user_text and self._is_vague(user_text) and state.clarify < 2:
            state.clarify += 1
            self._persist_state(state)
            LOGGER.debug("Clarification triggered for %s on attempt %s", sid, state.clarify)
            clarify_prompt = self._clarify_prompt(state)
            return {
                "next_utterance": clarify_prompt,
                "progress": {"index": state.index, "total": state.total},
                "risk_flag": False,
                "tts_text": None,
            }

        score_result = self._score_current_item(state, scoring_segments)
        if score_result:
            self._merge_scores(state, score_result["per_item_scores"])
            state.opinion = score_result.get("opinion") or state.opinion

        state.clarify = 0

        if state.index < state.total:
            state.index += 1
            item_id, question = self._current_item(state)
            self._persist_state(state)
            LOGGER.debug("Advancing %s to %s", sid, item_id)
            return {
                "next_utterance": question,
                "progress": {"index": state.index, "total": state.total},
                "risk_flag": False,
                "tts_text": None,
            }

        state.completed = True
        summary_payload = self._finalize_scores(sid, state)
        self._persist_state(state)
        return summary_payload

    # ------------------------------------------------------------------
    def _clarify_prompt(self, state: SessionState) -> str:
        item_id, _ = self._current_item(state)
        return CLARIFY_PROMPTS.get(item_id, DEFAULT_CLARIFY)

    def _current_item(self, state: SessionState) -> Tuple[str, str]:
        index = max(1, min(state.index, len(PHQ9)))
        return PHQ9[index - 1]

    def _prepare_segments(
        self, state: SessionState, segments: Optional[Iterable[Dict[str, object]]]
    ) -> List[Dict[str, object]]:
        prepared: List[Dict[str, object]] = []
        if not segments:
            return prepared

        for segment in segments:
            state.last_utt_index += 1
            normalized = dict(segment)
            normalized.setdefault("speaker", "patient")
            normalized.setdefault("utt_id", f"u{state.last_utt_index}")
            prepared.append(normalized)
        return prepared

    def _extract_user_text(self, segments: List[Dict[str, object]]) -> Optional[str]:
        for segment in reversed(segments):
            text = segment.get("text")
            if text:
                return str(text)
        return None

    def _check_risk(
        self,
        sid: str,
        state: SessionState,
        segments: List[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        for segment in segments:
            text = segment.get("text") or ""
            if not text:
                continue
            risk = risk_engine.evaluate(text)
            if risk.level == "high":
                now = datetime.now(timezone.utc).isoformat()
                reason = "、".join(risk.triggers) if risk.triggers else "触发高风险关键词"
                payload = {
                    "ts": now,
                    "reason": reason,
                    "match_text": text,
                }
                try:
                    self.repo.push_risk_event_stream(sid, payload)
                except Exception:  # pragma: no cover - runtime guard
                    LOGGER.exception("Failed to push risk event to stream for %s", sid)
                if hasattr(self.repo, "push_risk_event"):
                    self.repo.push_risk_event(sid, payload)
                elif hasattr(self.repo, "append_risk_event"):
                    self.repo.append_risk_event(sid, payload)
                state.completed = True
                LOGGER.info("Risk detected for %s: %s", sid, risk.triggers)
                return {
                    "next_utterance": "我已经记录到紧急风险，请立即寻求当地紧急服务帮助。",
                    "progress": {"index": state.index, "total": state.total},
                    "risk_flag": True,
                    "tts_text": None,
                    "risk": {"level": risk.level, "triggers": risk.triggers},
                }
        return None

    def _score_current_item(
        self,
        state: SessionState,
        transcripts: List[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        if not transcripts:
            return None

        item_id, question = self._current_item(state)
        latest_segment = next(
            (seg for seg in reversed(transcripts) if seg.get("speaker") == "patient"),
            transcripts[-1],
        )
        text = str(latest_segment.get("text", ""))
        score = self._rule_based_score(text)
        evidence_refs = [latest_segment.get("utt_id", "")]

        per_item_score = {
            "item_id": item_id,
            "name": question,
            "question": question,
            "score": score,
            "evidence_refs": evidence_refs,
        }

        opinion = self._generate_opinion(state.scores_acc, per_item_score)

        return {
            "per_item_scores": [per_item_score],
            "opinion": opinion,
        }

    def _rule_based_score(self, text: str) -> int:
        normalized = text.strip()
        lowered = normalized.lower()
        for keyword in NEGATION_KEYWORDS:
            if keyword in normalized:
                return 0
        for score, keywords in sorted(SCORE_KEYWORDS.items(), reverse=True):
            for keyword in keywords:
                if keyword in normalized or keyword in lowered:
                    return score
        return 1 if normalized else 0

    def _merge_scores(
        self, state: SessionState, new_scores: List[Dict[str, object]]
    ) -> None:
        scores_by_id = {score["item_id"]: score for score in state.scores_acc}
        for score in new_scores:
            item_id = score["item_id"]
            existing = scores_by_id.get(item_id)
            if existing is None or score.get("score", 0) >= existing.get("score", 0):
                scores_by_id[item_id] = score
        state.scores_acc = [scores_by_id[item_id] for item_id, _ in PHQ9 if item_id in scores_by_id]

    def _finalize_scores(self, sid: str, state: SessionState) -> Dict[str, object]:
        total_score = sum(int(score.get("score", 0)) for score in state.scores_acc)
        opinion = state.opinion or self._opinion_from_total(total_score)
        payload = {
            "per_item_scores": state.scores_acc,
            "total_score": total_score,
            "opinion": opinion,
        }
        try:
            repository.save_scores(sid, payload)
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to persist scores for %s", sid)

        next_utterance = "PHQ-9 评估完成，可导出报告。"
        return {
            "next_utterance": next_utterance,
            "progress": {"index": state.index, "total": state.total},
            "risk_flag": False,
            "tts_text": None,
        }

    def _complete_payload(self, state: SessionState, message: str) -> Dict[str, object]:
        return {
            "next_utterance": message,
            "progress": {"index": state.index, "total": state.total},
            "risk_flag": False,
            "tts_text": None,
        }

    def _generate_opinion(
        self, existing_scores: List[Dict[str, object]], new_score: Dict[str, object]
    ) -> str:
        temp_scores = {score["item_id"]: score for score in existing_scores}
        temp_scores[new_score["item_id"]] = new_score
        total = sum(int(score.get("score", 0)) for score in temp_scores.values())
        return self._opinion_from_total(total)

    @staticmethod
    def _opinion_from_total(total: int) -> str:
        if total >= 20:
            return "当前症状总分较高，建议尽快寻求专业帮助。"
        if total >= 15:
            return "存在中重度情绪困扰，建议尽快与专业人士沟通。"
        if total >= 10:
            return "存在一定程度抑郁症状，建议进一步评估。"
        if total >= 5:
            return "有轻度情绪困扰迹象，保持关注自身状态。"
        return "目前总分较低，继续保持良好状态。"

    def _load_state(self, sid: str) -> SessionState:
        raw = self.repo.load_session_state(sid) or {}
        state = SessionState(sid=sid)
        state.index = int(raw.get("index", state.index))
        state.total = int(raw.get("total", state.total)) or len(PHQ9)
        state.clarify = int(raw.get("clarify", state.clarify))
        state.completed = bool(raw.get("completed", state.completed))
        state.last_utt_index = int(raw.get("last_utt_index", state.last_utt_index))
        state.opinion = raw.get("opinion", state.opinion)
        state.scores_acc = raw.get("scores_acc", state.scores_acc)
        return state

    def _persist_state(self, state: SessionState) -> None:
        self.repo.save_session_state(
            state.sid,
            {
                "index": state.index,
                "total": state.total,
                "clarify": state.clarify,
                "scores_acc": state.scores_acc,
                "completed": state.completed,
                "last_utt_index": state.last_utt_index,
                "opinion": state.opinion,
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

    @staticmethod
    def _is_vague(text: str) -> bool:
        if not text:
            return False
        normalized = re.sub(r"[\s\W]+", "", text, flags=re.UNICODE).lower()
        for phrase in VAGUE_PHRASES:
            phrase_norm = re.sub(r"[\s\W]+", "", phrase, flags=re.UNICODE).lower()
            if phrase_norm and phrase_norm in normalized:
                return True
        return False


orchestrator = LangGraphMini()
