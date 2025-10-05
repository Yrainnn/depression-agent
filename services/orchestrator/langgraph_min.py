from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from packages.common.config import settings
from services.audio.asr_adapter import AsrError, StubASR, TingwuClientASR
from services.llm.json_client import ControllerDecision, DeepSeekJSONClient, HAMDResult
from services.llm.prompts import (
    get_prompt_hamd17,
    get_prompt_diagnosis,
    get_prompt_mdd_judgment,
)
from services.orchestrator.gap_utils import GAP_LABELS, detect_information_gaps
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
    scores_acc: List[Dict[str, Any]] = field(default_factory=list)
    completed: bool = False
    last_utt_index: int = 0
    opinion: Optional[str] = None
    last_text: str = ""
    analysis: Optional[Dict[str, Any]] = None
    controller_notice_logged: bool = False
    controller_unusable_turn: Optional[int] = None


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
        self.deepseek = DeepSeekJSONClient()
        self.prompt_diagnosis = get_prompt_diagnosis
        self.prompt_mdd = get_prompt_mdd_judgment
        self.CLARIFY_FALLBACKS = {
            "频次": "这种情况大概一周发生几次？",
            "持续时间": "每次大约持续多长时间？",
            "严重程度": "这对你的日常影响有多大？",
            "是否否定": "最近两周是否基本没有这种情况？",
            "是否有计划": "是否有具体计划或准备过相关工具？",
            "安全保障": "现在身边是否有人陪伴，能保证你的安全？",
        }
        self.ITEM_NAMES = {
            1: "抑郁情绪",
            2: "有罪感",
            3: "自杀倾向",
            4: "入睡困难",
            5: "睡眠维持障碍",
            6: "早醒",
            7: "工作和兴趣",
            8: "精神运动迟缓",
            9: "日夜症状变化",
            10: "精神性焦虑",
            11: "躯体性焦虑",
            12: "胃肠道症状",
            13: "全身症状",
            14: "性症状",
            15: "疑病倾向",
            16: "体重减轻",
            17: "自知力",
        }

    # ------------------------------------------------------------------
    def ask(self, sid: str) -> Dict[str, object]:
        state = self._load_state(sid)
        if state.completed:
            return self._complete_payload(state, COMPLETION_TEXT)

        item_id = self._current_item_id(state)
        question = pick_primary(item_id)
        return self._make_response(
            sid,
            state,
            question,
            turn_type="ask",
        )

    def step(
        self,
        sid: str,
        role: str,
        text: Optional[str] = None,
        audio_ref: Optional[str] = None,
        scale: str = "HAMD17",
    ) -> Dict[str, object]:
        state = self._load_state(sid)
        LOGGER.debug("Loaded state for %s: %s", sid, state)

        if state.completed:
            return self._complete_payload(state, COMPLETION_TEXT)

        if not text and not audio_ref:
            item = get_first_item()
            question = pick_primary(item)
            response = self._make_response(
                sid,
                state,
                question,
                turn_type="ask",
            )
            response.setdefault("risk", None)
            response.setdefault("analysis", state.analysis)
            return response

        raw_segments: List[Dict[str, Any]] = []
        if text:
            raw_segments.extend(self.stub_asr.transcribe(text=text))

        if audio_ref:
            try:
                raw_segments.extend(self.asr.transcribe(text=None, audio_ref=audio_ref))
            except AsrError as exc:
                LOGGER.warning("ASR audio transcription failed for %s: %s", sid, exc)
                if not raw_segments:
                    return self._make_response(
                        sid,
                        state,
                        MISSING_INPUT_PROMPT,
                        turn_type="clarify",
                    )

        prepared_segments = self._prepare_segments(state, raw_segments)
        for segment in prepared_segments:
            self.repo.append_transcript(sid, segment)
            LOGGER.debug("Appended transcript for %s: %s", sid, segment)

        transcripts = self.repo.get_transcripts(sid)

        risk_payload = self._check_risk(sid, state, prepared_segments, transcripts)
        if risk_payload is not None:
            return risk_payload

        user_text = self._extract_user_text(prepared_segments)
        if user_text:
            state.last_text = user_text

        item_id = self._current_item_id(state)
        scoring_segments = self._latest_segments(
            transcripts, self.window_n, self.window_seconds
        )

        dialogue_payload = self._build_dialogue_payload(sid)
        current_progress = {"index": item_id, "total": TOTAL_ITEMS}

        controller_enabled = settings.ENABLE_DS_CONTROLLER and self.deepseek.enabled()

        if not controller_enabled:
            if not state.controller_notice_logged:
                reason = (
                    "disabled via settings"
                    if not settings.ENABLE_DS_CONTROLLER
                    else "client not configured"
                )
                LOGGER.info("DeepSeek controller unavailable for %s: %s", sid, reason)
                state.controller_notice_logged = True
                self._persist_state(state)
            return self._fallback_flow(
                sid=sid,
                state=state,
                item_id=item_id,
                scoring_segments=scoring_segments,
                dialogue=dialogue_payload,
                transcripts=transcripts,
                user_text=user_text,
            )

        if (
            state.controller_unusable_turn is not None
            and state.controller_unusable_turn == state.last_utt_index
        ):
            LOGGER.debug(
                "DeepSeek controller temporarily sidelined for %s on turn %s",
                sid,
                state.controller_unusable_turn,
            )
            return self._fallback_flow(
                sid=sid,
                state=state,
                item_id=item_id,
                scoring_segments=scoring_segments,
                dialogue=dialogue_payload,
                transcripts=transcripts,
                user_text=user_text,
            )

        decision: Optional[ControllerDecision] = None
        try:
            decision = self.deepseek.plan_turn(dialogue_payload, current_progress)
        except Exception as exc:  # pragma: no cover - runtime guard
            log_method = LOGGER.warning
            if state.controller_notice_logged:
                log_method = LOGGER.debug
            log_method("DeepSeek controller planning failed for %s: %s", sid, exc)
            state.controller_notice_logged = True
            state.controller_unusable_turn = state.last_utt_index
            self._persist_state(state)
            return self._fallback_flow(
                sid=sid,
                state=state,
                item_id=item_id,
                scoring_segments=scoring_segments,
                dialogue=dialogue_payload,
                transcripts=transcripts,
                user_text=user_text,
            )

        if decision and decision.hamd_partial:
            analysis_payload = decision.hamd_partial.model_dump()
            state.analysis = analysis_payload
            try:
                self.repo.merge_scores(sid, analysis_payload)
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception("Failed to merge partial HAMD scores for %s", sid)
            items_payload = analysis_payload.get("items")
            if isinstance(items_payload, list):
                state.scores_acc = items_payload
        else:
            analysis_payload = None

        if not decision:
            state.controller_unusable_turn = state.last_utt_index
            self._persist_state(state)
            return self._fallback_flow(
                sid=sid,
                state=state,
                item_id=item_id,
                scoring_segments=scoring_segments,
                dialogue=dialogue_payload,
                transcripts=transcripts,
                user_text=user_text,
            )

        if analysis_payload:
            state.analysis = analysis_payload

        if state.controller_unusable_turn is not None:
            state.controller_unusable_turn = None
            self._persist_state(state)

        extra: Dict[str, Any] = {}
        if state.analysis:
            extra["analysis"] = state.analysis

        next_utt = decision.next_utterance or "请继续描述。"
        forced_target: Optional[int] = None
        try:
            last_clarify = self.repo.get_last_clarify_need(sid)
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to load last clarify target for %s", sid)
            last_clarify = None

        if decision.action == "clarify" and last_clarify and (user_text or prepared_segments):
            LOGGER.debug("Clarify override triggered for %s after user response", sid)
            try:
                self.repo.clear_last_clarify_need(sid)
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception("Failed to clear clarify target for %s", sid)
            forced_target = get_next_item(item_id)
            if forced_target == -1:
                decision_action = "finish"
                next_utt = COMPLETION_TEXT
            else:
                decision_action = "ask"
                next_utt = pick_primary(forced_target)
        else:
            decision_action = decision.action

        if decision_action == "clarify":
            if decision.clarify_target:
                try:
                    self.repo.set_last_clarify_need(
                        sid,
                        decision.clarify_target.item_id,
                        decision.clarify_target.clarify_need or "",
                    )
                except Exception:  # pragma: no cover - runtime guard
                    LOGGER.exception("Failed to persist clarify target for %s", sid)
            state.clarify += 1
            self._append_turn(
                sid,
                state,
                role="assistant",
                turn_type="clarify",
                text=next_utt,
            )
            return self._make_response(
                sid,
                state,
                next_utt,
                turn_type="clarify",
                extra=extra,
                record=False,
            )

        if decision_action == "ask":
            target_item = forced_target or decision.current_item_id
            if target_item in (None, 0):
                target_item = get_next_item(item_id)
            self._advance_to(sid, target_item or item_id, state)
            state.clarify = 0
            try:
                self.repo.clear_last_clarify_need(sid)
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception("Failed to clear clarify target for %s", sid)
            self._append_turn(
                sid,
                state,
                role="assistant",
                turn_type="ask",
                text=next_utt,
            )
            return self._make_response(
                sid,
                state,
                next_utt,
                turn_type="ask",
                extra=extra,
                record=False,
            )

        state.completed = True
        state.index = TOTAL_ITEMS
        self._persist_state(state)
        try:
            self.repo.mark_finished(sid)
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to mark session %s finished", sid)
        try:
            self.repo.clear_last_clarify_need(sid)
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to clear clarify target for %s", sid)
        completion_text = next_utt or COMPLETION_TEXT
        self._append_turn(
            sid,
            state,
            role="assistant",
            turn_type="ask",
            text=completion_text,
        )
        return self._make_response(
            sid,
            state,
            completion_text,
            turn_type="complete",
            extra=extra,
            record=False,
        )

    # ------------------------------------------------------------------
    def _make_response(
        self,
        sid: str,
        state: SessionState,
        text: str,
        *,
        risk_flag: bool = False,
        turn_type: str = "ask",
        extra: Optional[Dict[str, Any]] = None,
        record: bool = True,
    ) -> Dict[str, Any]:
        if record:
            self._record_assistant_turn(sid, state, text, turn_type)
        transcripts = self.repo.get_transcripts(sid)
        tts_url = self._make_tts(sid, text)
        previews = [
            seg.get("text")
            for seg in (transcripts[-2:] if transcripts else [])
            if seg.get("text")
        ]
        payload: Dict[str, Any] = {
            "next_utterance": text,
            "progress": {"index": min(state.index, state.total), "total": state.total},
            "risk_flag": risk_flag,
            "tts_text": text,
            "tts_url": tts_url or None,
        }
        if previews:
            payload["segments_previews"] = previews
        payload.setdefault("risk", None)
        payload.setdefault("analysis", state.analysis)
        if extra:
            payload.update(extra)
        return payload

    def _append_turn(
        self,
        sid: str,
        state: SessionState,
        *,
        role: str,
        turn_type: str,
        text: str,
    ) -> None:
        state.last_utt_index += 1
        event = {
            "utt_id": ("a" if role == "assistant" else "u")
            + str(state.last_utt_index),
            "text": text,
            "speaker": "assistant" if role == "assistant" else "patient",
            "role": role,
            "type": turn_type,
            "ts": [0, 0],
        }
        self.repo.append_transcript(sid, event)
        self._persist_state(state)

    def _record_assistant_turn(
        self, sid: str, state: SessionState, text: str, turn_type: str
    ) -> None:
        try:
            self._append_turn(
                sid,
                state,
                role="assistant",
                turn_type=turn_type,
                text=text,
            )
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to record assistant turn for %s", sid)

    def _advance_to(
        self, sid: str, item_id: int, state: Optional[SessionState] = None
    ) -> SessionState:
        target = max(get_first_item(), min(int(item_id), TOTAL_ITEMS))
        if state is None:
            state = self._load_state(sid)
        state.index = target
        state.clarify = 0
        self._persist_state(state)
        return state

    def _current_item_id(self, state: SessionState) -> int:
        return max(get_first_item(), min(state.index, TOTAL_ITEMS))

    def _prepare_segments(
        self, state: SessionState, segments: Optional[Iterable[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        if not segments:
            return prepared

        for segment in segments:
            state.last_utt_index += 1
            normalized = dict(segment)
            normalized.setdefault("speaker", "patient")
            normalized.setdefault("role", "user")
            normalized.setdefault("type", "answer")
            normalized.setdefault("utt_id", f"u{state.last_utt_index}")
            prepared.append(normalized)
        self._persist_state(state)
        return prepared

    def _extract_user_text(self, segments: List[Dict[str, Any]]) -> Optional[str]:
        for segment in reversed(segments):
            if segment.get("role") == "user" or segment.get("speaker") == "patient":
                text = segment.get("text")
                if text:
                    return str(text)
        return None

    def _check_risk(
        self,
        sid: str,
        state: SessionState,
        segments: List[Dict[str, Any]],
        transcripts: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        for segment in segments:
            if segment.get("role") not in {None, "user"}:
                continue
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
                extra = {"risk": {"level": risk.level, "triggers": risk.triggers}}
                return self._make_response(
                    sid,
                    state,
                    SAFE_RISK_TEXT,
                    risk_flag=True,
                    turn_type="risk",
                    extra=extra,
                )
        return None

    def _score_current_item(
        self,
        state: SessionState,
        transcripts: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not transcripts:
            return None

        item_id = self._current_item_id(state)
        question = pick_primary(item_id)
        latest_segment = next(
            (
                seg
                for seg in reversed(transcripts)
                if seg.get("speaker") == "patient" or seg.get("role") == "user"
            ),
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

    def _merge_scores(self, state: SessionState, new_scores: List[Dict[str, Any]]) -> None:
        scores_by_id = {score["item_id"]: score for score in state.scores_acc}
        for score in new_scores:
            item_id = score["item_id"]
            existing = scores_by_id.get(item_id)
            if existing is None or score.get("score", 0) >= existing.get("score", 0):
                scores_by_id[item_id] = score
        ordered: List[Dict[str, Any]] = []
        for idx in range(1, TOTAL_ITEMS + 1):
            key = f"H{idx:02d}"
            if key in scores_by_id:
                ordered.append(scores_by_id[key])
        state.scores_acc = ordered

    def _finalize_scores(
        self,
        sid: str,
        state: SessionState,
        transcripts: List[Dict[str, Any]],
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if state.analysis:
            payload.update(state.analysis)
            payload.setdefault("items", payload.get("items", payload.get("per_item_scores", [])))
            try:
                repository.save_scores(sid, payload)
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception("Failed to persist analysis scores for %s", sid)
        else:
            total_score = sum(int(score.get("score", 0)) for score in state.scores_acc)
            opinion = state.opinion or self._opinion_from_total(total_score)
            payload = {
                "per_item_scores": state.scores_acc,
                "total_score": {
                    "pre_correction_total": total_score,
                    "corrected_total": total_score,
                },
                "opinion": opinion,
            }
            try:
                repository.save_scores(sid, payload)
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception("Failed to persist scores for %s", sid)
        combined_extra = dict(extra or {})
        combined_extra.setdefault("analysis", state.analysis)
        response = self._make_response(
            sid,
            state,
            COMPLETION_TEXT,
            turn_type="complete",
            extra=combined_extra,
        )
        response.update(payload)
        return response

    def _complete_payload(self, state: SessionState, message: str) -> Dict[str, Any]:
        return self._make_response(
            state.sid,
            state,
            message,
            turn_type="complete",
        )

    def _detect_gaps(self, state: SessionState, item_id: int) -> List[str]:
        return detect_information_gaps(state.last_text, item_id=item_id)

    def _fallback_flow(
        self,
        *,
        sid: str,
        state: SessionState,
        item_id: int,
        scoring_segments: List[Dict[str, Any]],
        dialogue: List[Dict[str, Any]],
        transcripts: List[Dict[str, Any]],
        user_text: Optional[str],
    ) -> Dict[str, Any]:
        analysis_result = self._run_deepseek_analysis(dialogue)
        extra_payload: Dict[str, Any] = {}

        if analysis_result:
            analysis_dict = analysis_result.model_dump()
            state.analysis = analysis_dict
            extra_payload["analysis"] = analysis_dict
            self._store_analysis_scores(sid, state, analysis_result)
        else:
            state.analysis = None
            score_result = self._score_current_item(state, scoring_segments)
            if score_result:
                self._merge_scores(state, score_result["per_item_scores"])
                state.opinion = score_result.get("opinion") or state.opinion

        reverse_gap_labels = {label: key for key, label in GAP_LABELS.items()}
        fallback_gaps = self._detect_gaps(state, item_id)

        stored_gap_key: Optional[str] = None
        try:
            last_clarify = self.repo.get_last_clarify_need(sid)
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to read last clarify target for %s", sid)
            last_clarify = None

        if last_clarify and last_clarify.get("item_id") == item_id:
            stored_need = last_clarify.get("need")
            if isinstance(stored_need, str):
                stored_gap_key = reverse_gap_labels.get(stored_need, stored_need)
            if stored_gap_key and stored_gap_key not in fallback_gaps:
                try:
                    self.repo.clear_last_clarify_need(sid)
                except Exception:  # pragma: no cover - runtime guard
                    LOGGER.exception("Failed to clear clarify target for %s", sid)
                stored_gap_key = None

        if not analysis_result and user_text and state.clarify < 2:
            if self._is_vague(user_text) or fallback_gaps:
                state.clarify += 1
                clarify_key = fallback_gaps[0] if fallback_gaps else "severity"
                clarify_label = GAP_LABELS.get(clarify_key, clarify_key)
                try:
                    self.repo.set_last_clarify_need(sid, item_id, clarify_label)
                except Exception:  # pragma: no cover - runtime guard
                    LOGGER.exception("Failed to persist clarify target for %s", sid)
                clarify_prompt = pick_clarify(item_id, clarify_key)
                self._persist_state(state)
                return self._make_response(
                    sid,
                    state,
                    clarify_prompt,
                    turn_type="clarify",
                    extra=extra_payload,
                )

        clarify_question = None
        if analysis_result and user_text and state.clarify < 2:
            clarify_question = self._clarify_from_analysis(
                state, analysis_result, dialogue
            )

        if clarify_question:
            state.clarify += 1
            self._persist_state(state)
            return self._make_response(
                sid,
                state,
                clarify_question,
                turn_type="clarify",
                extra=extra_payload,
            )

        state.clarify = 0

        next_item = get_next_item(item_id)
        if next_item != -1:
            state.index = next_item
            self._persist_state(state)
            try:
                self.repo.clear_last_clarify_need(sid)
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception(
                    "Failed to clear clarify target after advancing for %s", sid
                )
            next_question = pick_primary(next_item)
            return self._make_response(
                sid,
                state,
                next_question,
                turn_type="ask",
                extra=extra_payload,
            )

        state.completed = True
        state.index = TOTAL_ITEMS
        self._persist_state(state)
        summary_payload = self._finalize_scores(
            sid, state, transcripts, extra=extra_payload
        )
        return summary_payload

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
        state.analysis = raw.get("analysis", state.analysis)
        state.controller_notice_logged = bool(
            raw.get("controller_notice_logged", state.controller_notice_logged)
        )
        controller_turn = raw.get("controller_unusable_turn")
        state.controller_unusable_turn = (
            int(controller_turn) if controller_turn is not None else None
        )
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
                "analysis": state.analysis,
                "controller_notice_logged": state.controller_notice_logged,
                "controller_unusable_turn": state.controller_unusable_turn,
            },
        )

    def _latest_segments(
        self,
        transcripts: List[Dict[str, Any]],
        max_items: int,
        max_seconds: int,
    ) -> List[Dict[str, Any]]:
        if not transcripts:
            return []

        if max_items <= 0:
            max_items = len(transcripts)

        cutoff: Optional[float] = None
        if max_seconds > 0:
            last_end = self._segment_end(transcripts[-1])
            if last_end is not None:
                cutoff = last_end - float(max_seconds)

        window: List[Dict[str, Any]] = []
        for segment in reversed(transcripts):
            if len(window) >= max_items:
                break

            end_time = self._segment_end(segment)
            if cutoff is not None and end_time is not None and end_time < cutoff:
                break

            window.append(segment)

        return list(reversed(window))

    @staticmethod
    def _segment_end(segment: Dict[str, Any]) -> Optional[float]:
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
        self, existing_scores: List[Dict[str, Any]], new_score: Dict[str, Any]
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
            return "情绪困扰程度为轻中度，建议持续关注并尝试自我调节。"
        if total >= 6:
            return "出现一定情绪波动，可继续观察并保持健康习惯。"
        return "当前情绪评分较低，如有需要仍可与专业人士交流。"

    def _build_dialogue_payload(self, sid: str) -> List[Dict[str, Any]]:
        dialogue: List[Dict[str, Any]] = []
        for segment in self.repo.get_transcripts(sid):
            role = segment.get("role")
            if not role:
                role = "assistant" if segment.get("speaker") != "patient" else "user"
            entry = {
                "sid": sid,
                "utt_id": segment.get("utt_id"),
                "role": role,
                "type": segment.get("type") or ("ask" if role == "assistant" else "answer"),
                "text": segment.get("text", ""),
                "ts": segment.get("ts") or [0, 0],
                "sentiment": segment.get("sentiment", "中性"),
            }
            dialogue.append(entry)
        return dialogue

    def _run_deepseek_analysis(self, dialogue: List[Dict[str, Any]]) -> Optional[HAMDResult]:
        if not dialogue or len(dialogue) < 4:
            return None
        if not self.deepseek.enabled():
            return None
        try:
            return self.deepseek.analyze(dialogue, get_prompt_hamd17())
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("DeepSeek analysis skipped: %s", exc)
            return None

    def _store_analysis_scores(
        self, sid: str, state: SessionState, result: HAMDResult
    ) -> None:
        items = []
        for item in result.items:
            items.append(
                {
                    "item_id": f"H{item.item_id:02d}",
                    "name": self.ITEM_NAMES.get(item.item_id, f"条目{item.item_id}"),
                    "question": pick_primary(item.item_id),
                    "score": item.score,
                    "max_score": MAX_SCORE.get(item.item_id, 4),
                    "evidence_refs": item.evidence_refs,
                    "score_type": item.score_type,
                    "score_reason": item.score_reason,
                    "clarify_need": item.clarify_need,
                }
            )
        state.scores_acc = items
        try:
            repository.save_scores(sid, result.model_dump())
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to persist DeepSeek result for %s", sid)
        self._persist_state(state)

    def _clarify_from_analysis(
        self,
        state: SessionState,
        result: HAMDResult,
        dialogue: List[Dict[str, Any]],
    ) -> Optional[str]:
        current_item = self._current_item_id(state)
        target = next(
            (
                item
                for item in result.items
                if item.item_id == current_item and item.score_type == "类型4" and item.clarify_need
            ),
            None,
        )
        if target is None:
            target = next(
                (item for item in result.items if item.score_type == "类型4" and item.clarify_need),
                None,
            )
        if target is None:
            return None
        clarify_need = target.clarify_need or ""
        evidence_text = "；".join(
            [entry.get("text", "") for entry in dialogue if entry.get("role") == "user"][-2:]
        )
        question = None
        if self.deepseek.enabled():
            question = self.deepseek.gen_clarify_question(
                target.item_id,
                self.ITEM_NAMES.get(target.item_id, f"条目{target.item_id}"),
                clarify_need,
                evidence_text,
            )
        if not question:
            question = self.CLARIFY_FALLBACKS.get(clarify_need, "能再具体说说这个情况吗？")
        return question


orchestrator = LangGraphMini()
