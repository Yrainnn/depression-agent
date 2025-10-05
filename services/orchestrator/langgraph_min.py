from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from packages.common.config import settings
from services.audio.asr_adapter import AsrError, StubASR, TingwuClientASR
from services.orchestrator.questions_hamd17 import (
    MAX_SCORE,
    get_first_item,
    get_next_item,
    pick_clarify,
    pick_primary,
)
from services.risk.engine import engine as risk_engine
from services.store.repository import repository
from services.tts.tts_adapter import TTSAdapter

LOGGER = logging.getLogger(__name__)

TOTAL_ITEMS = 17
SAFE_RISK_TEXT = "我已检测到较高风险，现在优先关注您的安全，请您保持在线并联系身边可求助的人。"
MISSING_INPUT_PROMPT = "未获取音频/文本，请重新描述一次好吗？"
COMPLETION_TEXT = "本次评估完成，感谢配合。稍后可下载报告。"

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


@dataclass
class SessionState:
    sid: str
    index: int = get_first_item()
    total: int = TOTAL_ITEMS
    clarify: int = 0
    scores_acc: List[Dict[str, object]] = field(default_factory=list)
    completed: bool = False
    last_utt_index: int = 0
    opinion: Optional[str] = None
    last_text: str = ""


class LangGraphMini:
    """Minimal LangGraph-inspired orchestrator implementing HAMD-17."""

    def __init__(self) -> None:
        self.repo = repository
        self.window_n = self._read_int_env("WINDOW_N", default=8)
        self.window_seconds = self._read_int_env("WINDOW_SECONDS", default=90)
        self.stub_asr = StubASR()
        try:
            self.asr = TingwuClientASR(settings)
        except AsrError as exc:  # pragma: no cover - configuration guard
            LOGGER.warning(
                "Failed to initialise TingWu client ASR, using stub instead: %s",
                exc,
            )
            self.asr = self.stub_asr
        self.tts = TTSAdapter()

    # ------------------------------------------------------------------
    def ask(self, sid: str) -> Dict[str, object]:
        state = self._load_state(sid)
        if state.completed:
            return self._complete_payload(state, COMPLETION_TEXT)

        item_id = self._current_item_id(state)
        question = pick_primary(item_id)
        self._persist_state(state)
        return self._make_response(
            sid,
            state,
            question,
            transcripts=[],
        )

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
            return self._complete_payload(state, COMPLETION_TEXT)

        if not text and not audio_ref and not segments:
            transcripts = self.repo.get_transcripts(sid)
            if not transcripts and state.index == get_first_item():
                question = pick_primary(self._current_item_id(state))
                return self._make_response(sid, state, question, transcripts=[])
            return self._make_response(
                sid,
                state,
                MISSING_INPUT_PROMPT,
                transcripts=transcripts,
            )

        raw_segments: List[Dict[str, object]] = []
        if segments is not None:
            raw_segments.extend(segments)
        else:
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
                        return self._make_response(
                            sid,
                            state,
                            MISSING_INPUT_PROMPT,
                            transcripts=self.repo.get_transcripts(sid),
                        )
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

        risk_payload = self._check_risk(sid, state, prepared_segments, transcripts)
        if risk_payload is not None:
            self._persist_state(state)
            return risk_payload

        user_text = self._extract_user_text(prepared_segments)
        if user_text:
            state.last_text = user_text

        item_id = self._current_item_id(state)
        gaps = self._detect_gaps(state, item_id)
        if user_text and state.clarify < 2 and (
            self._is_vague(user_text) or gaps
        ):
            state.clarify += 1
            clarify_key = gaps[0] if gaps else "severity"
            clarify_prompt = pick_clarify(item_id, clarify_key)
            self._persist_state(state)
            return self._make_response(
                sid,
                state,
                clarify_prompt,
                transcripts=transcripts,
            )

        score_result = self._score_current_item(state, scoring_segments)
        if score_result:
            self._merge_scores(state, score_result["per_item_scores"])
            state.opinion = score_result.get("opinion") or state.opinion

        state.clarify = 0

        next_item = get_next_item(item_id)
        if next_item != -1:
            state.index = next_item
            self._persist_state(state)
            next_question = pick_primary(next_item)
            return self._make_response(
                sid,
                state,
                next_question,
                transcripts=transcripts,
            )

        state.completed = True
        state.index = TOTAL_ITEMS
        summary_payload = self._finalize_scores(sid, state, transcripts)
        self._persist_state(state)
        return summary_payload

    # ------------------------------------------------------------------
    def _make_response(
        self,
        sid: str,
        state: SessionState,
        text: str,
        *,
        transcripts: Optional[List[Dict[str, object]]] = None,
        risk_flag: bool = False,
        extra: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        if transcripts is None:
            transcripts = self.repo.get_transcripts(sid)
        tts_url = self._make_tts(sid, text)
        previews = [
            seg.get("text")
            for seg in (transcripts[-2:] if transcripts else [])
            if seg.get("text")
        ]
        payload: Dict[str, object] = {
            "next_utterance": text,
            "progress": {"index": min(state.index, state.total), "total": state.total},
            "risk_flag": risk_flag,
            "tts_text": text,
            "tts_url": tts_url or None,
        }
        if previews:
            payload["segments_previews"] = previews
        if extra:
            payload.update(extra)
        return payload

    def _current_item_id(self, state: SessionState) -> int:
        return max(get_first_item(), min(state.index, TOTAL_ITEMS))

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
        transcripts: List[Dict[str, object]],
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
                state.index = TOTAL_ITEMS
                LOGGER.info("Risk detected for %s: %s", sid, risk.triggers)
                response = self._make_response(
                    sid,
                    state,
                    SAFE_RISK_TEXT,
                    transcripts=transcripts,
                    risk_flag=True,
                    extra={"risk": {"level": risk.level, "triggers": risk.triggers}},
                )
                return response
        return None

    def _score_current_item(
        self,
        state: SessionState,
        transcripts: List[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        if not transcripts:
            return None

        item_id = self._current_item_id(state)
        question = pick_primary(item_id)
        latest_segment = next(
            (seg for seg in reversed(transcripts) if seg.get("speaker") == "patient"),
            transcripts[-1],
        )
        text = str(latest_segment.get("text", ""))
        score = self._rule_based_score(text, item_id)
        evidence_refs = [latest_segment.get("utt_id", "")]

        per_item_score = {
            "item_id": f"H{item_id:02d}",
            "name": question,
            "question": question,
            "score": score,
            "max_score": MAX_SCORE.get(item_id, 4),
            "evidence_refs": evidence_refs,
        }

        opinion = self._generate_opinion(state.scores_acc, per_item_score)

        return {
            "per_item_scores": [per_item_score],
            "opinion": opinion,
        }

    def _rule_based_score(self, text: str, item_id: int) -> int:
        normalized = text.strip()
        lowered = normalized.lower()
        max_score = MAX_SCORE.get(item_id, 4)
        if not normalized:
            return 0
        if any(keyword in normalized for keyword in ["没有", "不", "很少", "没"]):
            return 0
        if any(keyword in normalized for keyword in ["严重", "完全", "一直", "难以"]):
            return min(4, max_score)
        if any(keyword in lowered for keyword in ["经常", "很多", "每天", "总是"]):
            return min(3, max_score)
        if any(keyword in lowered for keyword in ["有时", "偶尔", "有点", "几天"]):
            return min(2, max_score)
        return 1 if max_score >= 1 else 0

    def _merge_scores(self, state: SessionState, new_scores: List[Dict[str, object]]) -> None:
        scores_by_id = {score["item_id"]: score for score in state.scores_acc}
        for score in new_scores:
            item_id = score["item_id"]
            existing = scores_by_id.get(item_id)
            if existing is None or score.get("score", 0) >= existing.get("score", 0):
                scores_by_id[item_id] = score
        ordered: List[Dict[str, object]] = []
        for idx in range(1, TOTAL_ITEMS + 1):
            key = f"H{idx:02d}"
            if key in scores_by_id:
                ordered.append(scores_by_id[key])
        state.scores_acc = ordered

    def _finalize_scores(
        self, sid: str, state: SessionState, transcripts: List[Dict[str, object]]
    ) -> Dict[str, object]:
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

        response = self._make_response(
            sid,
            state,
            COMPLETION_TEXT,
            transcripts=transcripts,
        )
        response.update(payload)
        return response

    def _complete_payload(self, state: SessionState, message: str) -> Dict[str, object]:
        return self._make_response(state.sid, state, message, transcripts=self.repo.get_transcripts(state.sid))

    def _detect_gaps(self, state: SessionState, item_id: int) -> List[str]:
        text = (state.last_text or "").lower()
        gaps: List[str] = []
        if "次" not in text and "天" not in text and "每周" not in text:
            gaps.append("frequency")
        if "整天" not in text and "小时" not in text and "多久" not in text:
            gaps.append("duration")
        if "严重" not in text and "很难" not in text and "影响" not in text:
            gaps.append("severity")
        if "没有" in text or "不" in text:
            gaps.append("negation")
        if item_id == 3:
            if "计划" not in text:
                gaps.insert(0, "plan")
            if "安全" not in text and "陪伴" not in text:
                gaps.insert(0, "safety")
        return gaps

    def _make_tts(self, sid: str, text: str) -> str:
        try:
            return self.tts.synthesize(sid, text)
        except Exception:
            LOGGER.exception("TTS synthesis failed for %s", sid)
            return ""

    def _load_state(self, sid: str) -> SessionState:
        raw = self.repo.load_session_state(sid) or {}
        state = SessionState(sid=sid)
        state.index = int(raw.get("index", state.index))
        state.total = int(raw.get("total", state.total)) or TOTAL_ITEMS
        state.clarify = int(raw.get("clarify", state.clarify))
        state.completed = bool(raw.get("completed", state.completed))
        state.last_utt_index = int(raw.get("last_utt_index", state.last_utt_index))
        state.opinion = raw.get("opinion", state.opinion)
        state.scores_acc = raw.get("scores_acc", state.scores_acc)
        state.last_text = raw.get("last_text", state.last_text)
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
                "last_text": state.last_text,
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

    def _generate_opinion(
        self, existing_scores: List[Dict[str, object]], new_score: Dict[str, object]
    ) -> str:
        temp_scores = {score["item_id"]: score for score in existing_scores}
        temp_scores[new_score["item_id"]] = new_score
        total = sum(int(score.get("score", 0)) for score in temp_scores.values())
        return self._opinion_from_total(total)

    @staticmethod
    def _opinion_from_total(total: int) -> str:
        if total >= 25:
            return "当前症状总分较高，建议尽快寻求专业帮助。"
        if total >= 18:
            return "存在中度以上情绪困扰，建议与专业人士尽快沟通。"
        if total >= 12:
            return "存在一定程度的抑郁症状，建议继续评估与随访。"
        if total >= 7:
            return "有轻度情绪困扰迹象，建议保持关注并自我照护。"
        return "目前总分较低，可继续观察自身状态。"


orchestrator = LangGraphMini()
