from __future__ import annotations

import copy
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Dict, Iterable, List, Optional

from packages.common.config import settings
from services.audio.asr_adapter import AsrError, StubASR, TingwuClientASR
from services.llm.json_client import (
    ControllerDecision,
    DeepSeekJSONClient,
    DeepSeekTemporarilyUnavailableError,
    HAMDResult,
)
from services.llm.prompts import (
    get_prompt_hamd17,
    get_prompt_hamd17_controller,
    get_prompt_diagnosis,
    get_prompt_mdd_judgment,
)
from services.orchestrator.questions_hamd17 import (
    MAX_SCORE,
    get_first_item,
    get_next_item,
    pick_clarify,
    pick_primary,
)
from services.risk.engine import engine as risk_engine
from services.store.repository import repository
from services.report.build import build_pdf
from services.tts.tts_adapter import TTSAdapter

LOGGER = logging.getLogger(__name__)

TOTAL_ITEMS = 17
SAFE_RISK_TEXT = (
    "我已检测到较高风险。请先确保自身安全：联系家人/朋友或当地紧急热线。如您确认“已经安全/无需帮助”，我将继续评估。"
)
RISK_HOLD_REMINDER_TEXT = (
    "我注意到您可能仍处于风险关注阶段。如您已安全或无需立即帮助，请告诉我“已经安全/无需帮助”，我们就可以继续评估。"
)
MISSING_INPUT_PROMPT = "未获取音频/文本，请重新描述一次好吗？"
COMPLETION_TEXT = "本次评估完成，感谢配合。稍后可下载报告。"

REPORT_REQUEST_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"(获取|生成|下载|查看|打印|要|想要).{0,6}报告",
        r"报告.{0,6}(怎么|如何|咋|怎样).{0,4}(获取|生成|下载)",
        r"出一份报告",
        r"报告给我",
    ]
]

RISK_RELEASE_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"已经安全",
        r"已經安全",
        r"没事",
        r"沒事",
        r"无需帮助",
        r"無需幫助",
        r"不需要.{0,4}帮助",
        r"不用.{0,4}帮助",
        r"不需要紧急帮助",
        r"不需要立即帮助",
        r"没有自杀想法",
        r"沒有自殺想法",
        r"没有自杀念头",
        r"沒有自殺念頭",
        r"没有想自杀",
        r"沒有想自殺",
        r"不会伤害自己",
        r"不會傷害自己",
    ]
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
    def _has_asked_any_primary(self, sid: str) -> bool:
        """Return True if the session already contains an assistant 'ask' turn."""
        try:
            transcripts = self.repo.get_transcripts(sid)
        except Exception:  # pragma: no cover - defensive guard
            return False
        for segment in transcripts or []:
            role = segment.get("role") or segment.get("speaker")
            turn_type = segment.get("type")
            if role in {"assistant", "bot"} and turn_type == "ask":
                return True
        return False

    # ------------------------------------------------------------------
    def ask(self, sid: str) -> Dict[str, object]:
        state = self._load_state(sid)
        if state.completed:
            return self._complete_payload(state, COMPLETION_TEXT)

        item_id = self._current_item_id(state)
        transcripts = self.repo.get_transcripts(sid) or []
        dialogue = self._build_dialogue_payload(sid, transcripts)
        question = self._generate_primary_question(
            sid, state, item_id, transcripts, dialogue
        )
        return self._make_response(
            sid,
            state,
            question,
            turn_type="ask",
        )

    @staticmethod
    def _extract_controller_action(payload: Any) -> Optional[str]:
        if isinstance(payload, str):
            return payload or None
        if isinstance(payload, dict):
            for key in ("action", "decision", "type"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    return value
            # Some controllers may nest the actual action deeper.
            for key in ("decision", "result", "payload"):
                nested = payload.get(key)
                action = LangGraphMini._extract_controller_action(nested)
                if action:
                    return action
        return None

    @staticmethod
    def _extract_controller_question(payload: Any) -> Optional[str]:
        def _unwrap(value: Any) -> Optional[str]:
            if isinstance(value, str):
                text = value.strip()
                return text or None
            if isinstance(value, dict):
                for key in (
                    "next_utterance",
                    "question",
                    "utterance",
                    "primary_question",
                    "text",
                    "content",
                    "value",
                ):
                    candidate = _unwrap(value.get(key))
                    if candidate:
                        return candidate
                # Allow nested payloads such as {"data": {...}}
                for key in ("data", "result", "payload"):
                    candidate = _unwrap(value.get(key))
                    if candidate:
                        return candidate
                return None
            if isinstance(value, (list, tuple)):
                for item in value:
                    candidate = _unwrap(item)
                    if candidate:
                        return candidate
            return None

        if isinstance(payload, dict):
            # First try top-level fields.
            for key in (
                "next_utterance",
                "question",
                "primary_question",
                "utterance",
            ):
                candidate = _unwrap(payload.get(key))
                if candidate:
                    return candidate
            # Then inspect nested decision metadata.
            for key in ("decision", "data", "result", "payload"):
                candidate = _unwrap(payload.get(key))
                if candidate:
                    return candidate
        return _unwrap(payload)

    def _coerce_controller_decision(
        self, raw: Any, progress: Optional[Dict[str, Any]] = None
    ) -> Optional[ControllerDecision]:
        if raw is None:
            return None
        if isinstance(raw, ControllerDecision):
            return raw
        if not isinstance(raw, dict):
            LOGGER.debug("Unexpected controller payload type: %s", type(raw))
            return None

        data = dict(raw)
        decision_block = data.get("decision")
        action = self._extract_controller_action(data)
        if not action and isinstance(decision_block, dict):
            action = self._extract_controller_action(decision_block)
        if not action:
            action = "ask"

        question = self._extract_controller_question(data)
        if not question and isinstance(decision_block, dict):
            question = self._extract_controller_question(decision_block)

        data.setdefault("action", action)
        data.setdefault("decision", action)
        if question:
            data.setdefault("next_utterance", question)
            data.setdefault("question", question)
        elif "next_utterance" not in data:
            data["next_utterance"] = ""
        if "question" not in data:
            data["question"] = data.get("next_utterance")

        if "current_item_id" not in data and progress is not None:
            item_id = (
                (progress.get("index") if isinstance(progress, dict) else None) or 0
            )
            data["current_item_id"] = item_id

        clarify_prompt = data.get("clarify_prompt")
        if clarify_prompt and not data.get("clarify_target"):
            item_id = (
                data.get("clarify_item_id")
                or (progress.get("index") if progress else None)
                or 0
            )
            data["clarify_target"] = {"item_id": item_id, "clarify_need": clarify_prompt}

        try:
            return ControllerDecision.model_validate(data)
        except Exception as exc:
            LOGGER.debug("Failed to normalise controller decision: %s", exc)
            return None

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

        asked_primary = self._has_asked_any_primary(sid)
        item_id = self._current_item_id(state)
        print(f"🧠 进入 step(), 当前题号={item_id}", flush=True)

        if not text and not audio_ref:
            # 沿用当前条目，不要重置回首问
            transcripts = self.repo.get_transcripts(sid) or []
            dialogue = self._build_dialogue_payload(sid, transcripts)
            question = self._generate_primary_question(
                sid,
                state,
                item_id,
                transcripts,
                dialogue,
            )
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

        transcripts = self.repo.get_transcripts(sid) or []

        hold_payload = self._handle_risk_hold(sid, state, prepared_segments)
        if hold_payload is not None:
            return hold_payload

        risk_payload = self._check_risk(sid, state, prepared_segments, transcripts)
        if risk_payload is not None:
            return risk_payload

        user_text = self._extract_user_text(prepared_segments)
        if user_text:
            state.last_text = user_text

        report_payload = self._maybe_handle_report_request(sid, state, user_text)
        if report_payload is not None:
            return report_payload

        dialogue_payload = self._build_dialogue_payload(sid, transcripts)

        if not asked_primary:
            question = self._generate_primary_question(
                sid,
                state,
                self._current_item_id(state),
                transcripts,
                dialogue_payload,
            )
            return self._make_response(
                sid,
                state,
                question,
                turn_type="ask",
            )

        item_id = self._current_item_id(state)
        scoring_segments = self._latest_segments(
            transcripts, self.window_n, self.window_seconds
        )
        current_progress = {"index": item_id, "total": TOTAL_ITEMS}

        controller_enabled = (
            settings.ENABLE_DS_CONTROLLER and self.deepseek.usable()
        )

        if not controller_enabled:
            if not state.controller_notice_logged:
                reason = (
                    "disabled via settings"
                    if not settings.ENABLE_DS_CONTROLLER
                    else (
                        "client not configured"
                        if not self.deepseek.enabled()
                        else "temporarily unavailable"
                    )
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
            print(
                f"🧩 调用 DeepSeek 控制器生成第{item_id}题或澄清问",
                flush=True,
            )
            decision_payload = self.deepseek.plan_turn(
                dialogue_payload,
                current_progress,
                prompt=get_prompt_hamd17_controller(),
            )
            print(f"🤖 DeepSeek 返回: {decision_payload}", flush=True)

            # 🧠 统一字段兼容读取
            if isinstance(decision_payload, dict):
                decision_action = (
                    decision_payload.get("action")
                    or decision_payload.get("decision")
                    or decision_payload.get("type")
                )
                ask_text = (
                    decision_payload.get("next_utterance")
                    or decision_payload.get("question")
                )
            else:
                decision_action = None
                ask_text = None

            # 🩺 决策有效性检查
            if not decision_payload or not isinstance(decision_payload, dict):
                print("⚠️ DeepSeek 返回对象无效，启用 fallback。", flush=True)
                return self._fallback_flow(
                    sid=sid,
                    state=state,
                    item_id=item_id,
                    scoring_segments=scoring_segments,
                    dialogue=dialogue_payload,
                    transcripts=transcripts,
                    user_text=user_text,
                )

            normalized_payload = dict(decision_payload)

            if decision_action and ask_text:
                print(
                    f"✅ DeepSeek 主问有效，action={decision_action}, 问句={ask_text}",
                    flush=True,
                )
            elif decision_action == "ask" and not ask_text:
                print(
                    "⚠️ DeepSeek 决策有效但无输出，回退固定题库。",
                    flush=True,
                )
                ask_text = pick_primary(item_id)
            elif not decision_action:
                print(
                    "⚠️ DeepSeek 未返回有效动作字段，默认使用 ask。",
                    flush=True,
                )
                decision_action = "ask"

            if ask_text is None:
                ask_text = ""

            normalized_payload.setdefault("action", decision_action)
            normalized_payload.setdefault("decision", decision_action)
            normalized_payload.setdefault("next_utterance", ask_text)
            normalized_payload.setdefault("question", ask_text)

            decision = self._coerce_controller_decision(
                normalized_payload, current_progress
            )
            decision_payload = normalized_payload
        except DeepSeekTemporarilyUnavailableError as exc:
            LOGGER.debug("DeepSeek controller temporarily unavailable for %s: %s", sid, exc)
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

        if not decision:
            print("⚠️ DeepSeek 决策无效，启用 fallback 流程。", flush=True)
            return self._fallback_flow(
                sid=sid,
                state=state,
                item_id=item_id,
                scoring_segments=scoring_segments,
                dialogue=dialogue_payload,
                transcripts=transcripts,
                user_text=user_text,
            )

        if state.controller_unusable_turn is not None:
            state.controller_unusable_turn = None
            self._persist_state(state)

        self._apply_controller_scores(sid, state, item_id, decision)

        if state.scores_acc:
            state.analysis = self._analysis_from_scores(state)
        else:
            state.analysis = None

        controller_text = (decision.next_utterance or "").strip()
        if not controller_text:
            extracted_text = self._extract_controller_question(decision_payload)
            controller_text = (extracted_text or "").strip()

        decision_action = (decision.action or "ask").lower()
        print(
            f"🎯 控制器动作={decision_action}, 输出={controller_text or 'None'}",
            flush=True,
        )

        ask_target_override: Optional[int] = None

        clarify_target_id = decision.current_item_id or item_id
        clarify_prompt = ""
        if decision.clarify_target:
            clarify_target_id = decision.clarify_target.item_id or clarify_target_id
            clarify_prompt = decision.clarify_target.clarify_need or ""

        if decision_action == "clarify":
            if state.clarify >= 2:
                print("⚠️ 达到最大澄清次数上限，强制推进。", flush=True)
                ask_target = decision.current_item_id or get_next_item(item_id)
                try:
                    ask_target_int = int(ask_target)
                except (TypeError, ValueError):
                    ask_target_int = get_next_item(item_id)
                if ask_target_int in (None, -1):
                    decision_action = "finish"
                    controller_text = controller_text or ""
                else:
                    self._advance_to(sid, ask_target_int, state)
                    print(
                        f"📈 DeepSeek 决策推进至第 {state.index} 题。",
                        flush=True,
                    )
                    ask_target_override = state.index
                    print(
                        f"🔁 重新调用 DeepSeek 生成第{state.index}题主问",
                        flush=True,
                    )
                    followup_progress = {
                        "index": state.index,
                        "total": TOTAL_ITEMS,
                    }
                    try:
                        followup_payload = self.deepseek.plan_turn(
                            dialogue_payload,
                            followup_progress,
                            prompt=get_prompt_hamd17_controller(),
                        )
                        print(
                            f"🤖 DeepSeek 返回(重新调用): {followup_payload}",
                            flush=True,
                        )
                        followup_decision = self._coerce_controller_decision(
                            followup_payload, followup_progress
                        )
                        self._apply_controller_scores(
                            sid, state, state.index, followup_decision
                        )
                        followup_text = (
                            (followup_decision.next_utterance or "")
                            if followup_decision
                            else ""
                        ).strip()
                        if not followup_text:
                            extracted_followup = self._extract_controller_question(
                                followup_payload
                            )
                            followup_text = (extracted_followup or "").strip()
                        if followup_text:
                            controller_text = followup_text
                            decision_payload = followup_payload
                            if followup_decision:
                                decision = followup_decision
                        else:
                            print(
                                "⚠️ DeepSeek 无输出，回退固定问题。",
                                flush=True,
                            )
                            controller_text = pick_primary(state.index)
                    except Exception as exc:  # pragma: no cover - runtime guard
                        print(
                            f"⚠️ DeepSeek 生成第{state.index}题失败，回退固定题库: {exc}",
                            flush=True,
                        )
                        controller_text = pick_primary(state.index)
                    decision_action = "ask"
            else:
                state.clarify += 1
                print(
                    f"🗣️ DeepSeek 要求继续澄清，第 {state.clarify} 次。",
                    flush=True,
                )

        if state.analysis:
            extra: Dict[str, Any] = {"analysis": state.analysis}
        else:
            extra = {}

        if decision_action == "clarify":
            clarify_text = controller_text
            if not clarify_text:
                print("⚠️ DeepSeek 无有效输出，回退固定问题。", flush=True)
                clarify_text = pick_clarify(clarify_target_id, clarify_prompt)
            if not clarify_text:
                clarify_text = pick_clarify(item_id, clarify_prompt)
            try:
                prompt_to_store = clarify_prompt or clarify_text
                if prompt_to_store:
                    self.repo.set_last_clarify_need(
                        sid,
                        clarify_target_id or item_id,
                        prompt_to_store,
                    )
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception("Failed to persist clarify target for %s", sid)
            self._append_turn(
                sid,
                state,
                role="assistant",
                turn_type="clarify",
                text=clarify_text,
            )
            print(f"📢 最终输出问句: {clarify_text}", flush=True)
            return self._make_response(
                sid,
                state,
                clarify_text,
                turn_type="clarify",
                extra=extra,
                record=False,
            )

        if ask_target_override is not None:
            ask_target_int = ask_target_override
        else:
            ask_target = decision.current_item_id or get_next_item(item_id)
            try:
                ask_target_int = int(ask_target)
            except (TypeError, ValueError):
                ask_target_int = item_id
            if ask_target_int <= item_id:
                ask_target_int = get_next_item(item_id)

        if decision_action == "ask":
            if ask_target_int in (None, -1):
                decision_action = "finish"
            else:
                if ask_target_override is None:
                    self._advance_to(sid, ask_target_int, state)
                else:
                    state.index = ask_target_int
                    state.clarify = 0
                    try:
                        self.repo.clear_last_clarify_need(state.sid)
                    except Exception:  # pragma: no cover - runtime guard
                        LOGGER.exception(
                            "Failed to clear clarify target for %s", state.sid
                        )
                    self._persist_state(state)
                print(
                    f"📈 DeepSeek 决策推进至第 {state.index} 题。",
                    flush=True,
                )
                if not controller_text:
                    print("⚠️ DeepSeek 无有效输出，回退固定问题。", flush=True)
                    controller_text = pick_primary(state.index)

        if decision_action == "ask":
            self._append_turn(
                sid,
                state,
                role="assistant",
                turn_type="ask",
                text=controller_text,
            )
            print(f"📢 最终输出问句: {controller_text}", flush=True)
            return self._make_response(
                sid,
                state,
                controller_text,
                turn_type="ask",
                extra=extra,
                record=False,
            )

        state.completed = True
        state.index = TOTAL_ITEMS
        state.clarify = 0
        self._persist_state(state)
        try:
            self.repo.mark_finished(sid)
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to mark session %s finished", sid)
        try:
            self.repo.clear_last_clarify_need(sid)
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to clear clarify target for %s", sid)
        completion_text = controller_text or COMPLETION_TEXT
        self._append_turn(
            sid,
            state,
            role="assistant",
            turn_type="ask",
            text=completion_text,
        )
        print(f"📢 最终输出问句: {completion_text}", flush=True)
        return self._make_response(
            sid,
            state,
            completion_text,
            turn_type="complete",
            extra=extra,
            record=False,
        )

    def _apply_controller_scores(
        self,
        sid: str,
        state: SessionState,
        item_id: int,
        decision: Optional[ControllerDecision],
    ) -> None:
        if not decision or not decision.hamd_partial:
            return
        print("📊 检测到 HAMD 局部评分结果，开始合并...", flush=True)
        partial_payload = decision.hamd_partial.model_dump()
        try:
            self.repo.merge_scores(sid, partial_payload)
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to merge partial HAMD scores for %s", sid)
        items_payload = partial_payload.get("items") or []
        normalized_items = [
            entry
            for entry in (self._normalize_score_entry(item) for item in items_payload)
            if entry
        ]
        if normalized_items:
            self._merge_scores(state, normalized_items)
        total_payload = partial_payload.get("total_score") or {}
        total_value = self._extract_total_score(total_payload)
        if total_value is not None:
            print(
                f"✅ 当前题号={item_id} 局部评分={total_value}",
                flush=True,
            )
            state.opinion = self._opinion_from_total(total_value)

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
        payload["risk"] = payload.get("risk", None)
        # 优先使用 extra.analysis，其次 state.analysis
        if extra and "analysis" in extra:
            payload["analysis"] = copy.deepcopy(extra["analysis"])
        else:
            payload["analysis"] = copy.deepcopy(state.analysis) if state.analysis else None
        if extra:
            for key, value in extra.items():
                if key == "analysis":
                    continue
                payload[key] = value
        return payload

    def _generate_primary_question(
        self,
        sid: str,
        state: SessionState,
        item_id: int,
        transcripts: Optional[List[Dict[str, Any]]],
        dialogue: List[Dict[str, Any]],
    ) -> str:
        fallback = pick_primary(item_id)

        if not settings.ENABLE_DS_CONTROLLER or not self.deepseek.usable():
            return fallback

        print(f"🧠 调用 DeepSeek 控制器生成第{item_id}题", flush=True)
        print(
            f"ENABLE_DS_CONTROLLER={getattr(settings, 'ENABLE_DS_CONTROLLER', None)}",
            flush=True,
        )
        print(f"DeepSeek usable={self.deepseek.usable()}", flush=True)

        progress = {"index": item_id, "total": TOTAL_ITEMS}
        try:
            decision_payload = self.deepseek.plan_turn(
                dialogue,
                progress,
                prompt=get_prompt_hamd17_controller(),
            )
            decision = self._coerce_controller_decision(decision_payload, progress)
        except DeepSeekTemporarilyUnavailableError as exc:
            LOGGER.debug("DeepSeek question generation unavailable for %s: %s", sid, exc)
            state.controller_notice_logged = True
            self._persist_state(state)
            return fallback
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.info("DeepSeek question generation failed for %s: %s", sid, exc)
            print(f"❌ DeepSeek question generation error: {exc}", flush=True)
            state.controller_notice_logged = True
            self._persist_state(state)
            return fallback

        if decision and decision.next_utterance:
            question = (decision.next_utterance or "").strip()
            if question:
                return question

        fallback_from_payload = self._extract_controller_question(decision_payload)
        if fallback_from_payload:
            stripped = fallback_from_payload.strip()
            if stripped:
                LOGGER.debug(
                    "Using controller payload question fallback for %s (item %s)",
                    sid,
                    item_id,
                )
                return stripped

        LOGGER.debug(
            "DeepSeek question generation returned no utterance for %s (item %s)",
            sid,
            item_id,
        )
        return fallback

    def _maybe_handle_report_request(
        self, sid: str, state: SessionState, user_text: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not user_text:
            return None

        normalized = user_text.strip()
        if not normalized:
            return None

        lowered = normalized.lower()
        matched = any(
            pattern.search(normalized) or pattern.search(lowered)
            for pattern in REPORT_REQUEST_PATTERNS
        )
        if not matched:
            return None

        self._persist_state(state)

        score_payload = self._prepare_report_scores(sid, state)
        question = pick_primary(self._current_item_id(state))

        if not score_payload:
            message = f"当前暂无足够信息生成报告，我们先继续评估：{question}"
            return self._make_response(
                sid,
                state,
                message,
                turn_type="ask",
                extra={"report_generated": False},
            )

        try:
            report_result = build_pdf(sid, score_payload)
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to build report for %s: %s", sid, exc)
            message = f"生成报告时遇到问题，我们继续当前评估：{question}"
            return self._make_response(
                sid,
                state,
                message,
                turn_type="ask",
                extra={"report_generated": False},
            )

        report_url = (
            report_result.get("report_url")
            if isinstance(report_result, dict)
            else None
        )
        if not report_url:
            message = f"暂时无法提供下载链接，我们先继续评估：{question}"
            return self._make_response(
                sid,
                state,
                message,
                turn_type="ask",
                extra={"report_generated": False},
            )

        message = f"已为您生成评估报告，可通过此链接下载：{report_url}。我们继续：{question}"
        extra = {"report_generated": True, "report_url": report_url}
        if state.analysis:
            extra["analysis"] = state.analysis
        return self._make_response(
            sid,
            state,
            message,
            turn_type="ask",
            extra=extra,
        )

    def _prepare_report_scores(
        self, sid: str, state: SessionState
    ) -> Optional[Dict[str, Any]]:
        try:
            stored_scores = self.repo.load_scores(sid)
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to load stored scores for %s", sid)
            stored_scores = None

        payload: Dict[str, Any] = {}
        if isinstance(stored_scores, dict):
            payload.update(copy.deepcopy(stored_scores))
        elif isinstance(stored_scores, list):
            payload["items"] = copy.deepcopy(stored_scores)

        if state.scores_acc:
            payload.setdefault("items", copy.deepcopy(state.scores_acc))

        if state.analysis and isinstance(state.analysis, dict):
            total_block = state.analysis.get("total_score")
            if total_block:
                payload.setdefault("total_score", copy.deepcopy(total_block))

        if state.opinion:
            if isinstance(payload.get("opinion"), dict):
                payload["opinion"].setdefault("summary", state.opinion)
            else:
                payload["opinion"] = {"summary": state.opinion}

        if not payload.get("items") and not payload.get("per_item_scores"):
            return None

        return payload

    def _emit_risk_event(self, sid: str, payload: Dict[str, Any]) -> None:
        try:
            self.repo.push_risk_event_stream(sid, payload)
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to push risk event to stream for %s", sid)
        if hasattr(self.repo, "push_risk_event"):
            self.repo.push_risk_event(sid, payload)
        elif hasattr(self.repo, "append_risk_event"):
            self.repo.append_risk_event(sid, payload)

    def _handle_risk_hold(
        self, sid: str, state: SessionState, segments: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        risk_hold_active = getattr(state, "risk_hold", False)
        if not risk_hold_active:
            return None

        combined_text = "".join(
            str(segment.get("text") or "")
            for segment in segments
            if segment.get("role") in {None, "user"}
            or segment.get("speaker") == "patient"
        ).strip()

        if combined_text:
            release_hit = any(pattern.search(combined_text) for pattern in RISK_RELEASE_PATTERNS)
            risk_snapshot = risk_engine.evaluate(combined_text)
            if release_hit and risk_snapshot.level != "high":
                setattr(state, "risk_hold", False)
                release_payload: Dict[str, Any] = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "reason": "用户确认已安全，解除风险保持态",
                    "match_text": combined_text,
                    "phase": "release",
                    "risk": {
                        "level": "cleared",
                        "triggers": [],
                        "reason": "user_confirmed_safe",
                    },
                }
                self._emit_risk_event(sid, release_payload)
                self._persist_state(state)
                return None

        setattr(state, "risk_hold", True)
        self._persist_state(state)
        return self._make_response(
            sid,
            state,
            RISK_HOLD_REMINDER_TEXT,
            risk_flag=True,
            turn_type="risk",
            extra={"risk": {"level": "hold"}},
        )

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
        # 统一清理上一轮的 clarify 记录，避免遗留阻塞推进
        try:
            self.repo.clear_last_clarify_need(state.sid)
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to clear clarify target for %s", state.sid)
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
            if segment.get("role") not in {None, "user"} and segment.get("speaker") != "patient":
                continue
            raw_text = segment.get("text")
            if not raw_text:
                continue
            text = str(raw_text)
            risk = risk_engine.evaluate(text)
            if risk.level == "high":
                now = datetime.now(timezone.utc).isoformat()
                reason = risk.reason or (
                    "、".join(risk.triggers) if risk.triggers else "触发高风险关键词"
                )
                event_payload: Dict[str, Any] = {
                    "ts": now,
                    "reason": reason,
                    "match_text": text,
                    "phase": "hold",
                    "risk": {
                        "level": risk.level,
                        "triggers": risk.triggers,
                        "reason": risk.reason,
                    },
                }
                self._emit_risk_event(sid, event_payload)
                setattr(state, "risk_hold", True)
                self._persist_state(state)
                LOGGER.info("Risk detected for %s: %s", sid, risk.triggers)
                extra = {"risk": event_payload["risk"]}
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
        dialogue: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not transcripts:
            return None

        semantic_result = self._semantic_score_current_item(
            state, transcripts, dialogue
        )
        if semantic_result:
            return semantic_result

        return None

    def _semantic_score_current_item(
        self,
        state: SessionState,
        transcripts: List[Dict[str, Any]],
        dialogue: Optional[List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        if not self.deepseek.usable():
            return None

        dialogue_payload = list(dialogue) if dialogue else self._build_dialogue_payload(
            state.sid
        )
        if not dialogue_payload:
            return None

        try:
            result = self.deepseek.analyze(dialogue_payload, get_prompt_hamd17())
        except DeepSeekTemporarilyUnavailableError as exc:
            LOGGER.debug("DeepSeek semantic scoring skipped for %s: %s", state.sid, exc)
            return None
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.debug(
                "DeepSeek semantic scoring skipped for %s: %s", state.sid, exc
            )
            return None

        item_id = self._current_item_id(state)
        target = next((item for item in result.items if item.item_id == item_id), None)
        if target is None:
            return None

        question = pick_primary(item_id)
        latest_segment = next(
            (
                seg
                for seg in reversed(transcripts)
                if seg.get("speaker") == "patient" or seg.get("role") == "user"
            ),
            transcripts[-1],
        )
        evidence_refs = [ref for ref in target.evidence_refs if ref]
        if not evidence_refs and latest_segment:
            evidence_id = latest_segment.get("utt_id", "")
            if evidence_id:
                evidence_refs = [evidence_id]

        per_item_score: Dict[str, Any] = {
            "item_id": f"H{item_id:02d}",
            "name": question,
            "question": question,
            "score": min(int(target.score), MAX_SCORE.get(item_id, 4)),
            "max_score": MAX_SCORE.get(item_id, 4),
            "evidence_refs": evidence_refs,
            "score_type": target.score_type,
            "score_reason": target.score_reason,
            "dialogue_evidence": target.dialogue_evidence,
            "symptom_summary": target.symptom_summary,
            "clarify_need": target.clarify_need,
        }

        opinion = self._generate_opinion(state.scores_acc, per_item_score)

        return {
            "per_item_scores": [per_item_score],
            "opinion": opinion,
        }

    @staticmethod
    def _extract_total_score(payload: Dict[str, Any]) -> Optional[int]:
        for key in ("corrected_total", "pre_correction_total", "total"):
            value = payload.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _standardize_score_entry(
        self, item_id: int, payload: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        base = dict(payload or {})
        normalized = copy.deepcopy(base)
        normalized["item_id"] = f"H{item_id:02d}"
        normalized["name"] = normalized.get("name") or self.ITEM_NAMES.get(
            item_id, f"条目{item_id}"
        )
        normalized["question"] = normalized.get("question") or pick_primary(item_id)
        try:
            score_value = int(normalized.get("score", 0))
        except (TypeError, ValueError):
            score_value = 0
        max_score = MAX_SCORE.get(item_id, 4)
        normalized["score"] = max(0, min(score_value, max_score))
        normalized["max_score"] = max_score
        normalized["evidence_refs"] = list(normalized.get("evidence_refs") or [])
        normalized["dialogue_evidence"] = normalized.get("dialogue_evidence") or "直接引用"
        normalized["symptom_summary"] = normalized.get("symptom_summary") or ""
        normalized["score_type"] = normalized.get("score_type") or "类型1"
        normalized["score_reason"] = normalized.get("score_reason") or ""
        clarify_need = normalized.get("clarify_need")
        normalized["clarify_need"] = clarify_need if clarify_need not in {"", None} else None
        return normalized

    def _normalize_score_entry(
        self, payload: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        raw_id = payload.get("item_id")
        item_id: Optional[int]
        if isinstance(raw_id, int):
            item_id = raw_id
        elif isinstance(raw_id, str):
            stripped = raw_id.strip().upper()
            if stripped.startswith("H"):
                stripped = stripped[1:]
            if not stripped:
                return None
            try:
                item_id = int(stripped)
            except ValueError:
                return None
        else:
            try:
                item_id = int(raw_id)
            except (TypeError, ValueError):
                return None
        if not (1 <= item_id <= TOTAL_ITEMS):
            return None
        return self._standardize_score_entry(item_id, payload)

    def _merge_scores(self, state: SessionState, new_scores: List[Dict[str, Any]]) -> None:
        scores_by_id: Dict[str, Dict[str, Any]] = {}
        for existing in state.scores_acc:
            normalized = self._normalize_score_entry(existing)
            if not normalized:
                continue
            scores_by_id[normalized["item_id"]] = normalized

        def score_value(entry: Dict[str, Any]) -> int:
            try:
                return int(entry.get("score", 0))
            except (TypeError, ValueError):
                return 0

        for score in new_scores:
            normalized = self._normalize_score_entry(score)
            if not normalized:
                continue
            key = normalized["item_id"]
            current = scores_by_id.get(key)
            if current is None:
                scores_by_id[key] = normalized
                continue
            existing_value = score_value(current)
            candidate_value = score_value(normalized)
            if candidate_value > existing_value:
                scores_by_id[key] = normalized
            elif candidate_value == existing_value:
                current_refs = len(current.get("evidence_refs") or [])
                candidate_refs = len(normalized.get("evidence_refs") or [])
                if candidate_refs > current_refs:
                    scores_by_id[key] = normalized
                elif candidate_refs == current_refs:
                    current_summary = current.get("symptom_summary") or ""
                    candidate_summary = normalized.get("symptom_summary") or ""
                    if not current_summary and candidate_summary:
                        scores_by_id[key] = normalized

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

    def _analysis_from_scores(self, state: SessionState) -> Dict[str, Any]:
        """根据已有的 scores_acc 生成一个轻量级 analysis 快照。"""
        normalized_scores: List[Dict[str, Any]] = []
        for entry in state.scores_acc:
            normalized = self._normalize_score_entry(entry)
            if normalized:
                normalized_scores.append(normalized)
        state.scores_acc = normalized_scores

        scores_map = {score["item_id"]: score for score in normalized_scores}
        seq_list: List[str] = []
        items: List[Dict[str, Any]] = []
        total = 0
        type4_count = 0
        for idx in range(1, TOTAL_ITEMS + 1):
            key = f"H{idx:02d}"
            score_entry = scores_map.get(key)
            if score_entry:
                try:
                    score_value = int(score_entry.get("score", 0))
                except (TypeError, ValueError):
                    score_value = 0
            else:
                score_value = 0
            seq_list.append(str(score_value))
            total += score_value
            if not score_entry:
                continue
            if score_entry.get("score_type") == "类型4":
                type4_count += 1
            items.append(
                {
                    "item_id": idx,
                    "symptom_summary": score_entry.get("symptom_summary")
                    or self.ITEM_NAMES.get(idx, f"条目{idx}"),
                    "dialogue_evidence": score_entry.get("dialogue_evidence", "直接引用"),
                    "evidence_refs": score_entry.get("evidence_refs", []),
                    "score": score_value,
                    "score_type": score_entry.get("score_type", "类型1"),
                    "score_reason": score_entry.get("score_reason", ""),
                    "clarify_need": score_entry.get("clarify_need"),
                }
            )

        avg = total / type4_count if type4_count else 0
        correction_basis = (
            f"类型4条目数量N4={type4_count}，平均分X={avg:.2f}，修正总分=A+X×N4={total}"
        )

        return {
            "items": items,
            "total_score": {
                "得分序列": ",".join(seq_list),
                "pre_correction_total": total,
                "corrected_total": total,
                "correction_basis": correction_basis,
            },
        }

    def _complete_payload(self, state: SessionState, message: str) -> Dict[str, Any]:
        return self._make_response(
            state.sid,
            state,
            message,
            turn_type="complete",
        )

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
        analysis_result = self._run_ds_analysis_stream(dialogue)
        extra_payload: Dict[str, Any] = {}
        clarify_prompt: Optional[str] = None
        clarify_item_id: Optional[int] = None

        if analysis_result:
            self._store_analysis_scores(sid, state, analysis_result)
            target_item = next(
                (item for item in analysis_result.items if item.item_id == item_id),
                None,
            )
            if target_item and target_item.clarify_need:
                clarify_item_id = target_item.item_id
                clarify_prompt = target_item.clarify_prompt or pick_clarify(
                    target_item.item_id, ""
                )
        else:
            score_result = self._score_current_item(state, scoring_segments, dialogue)
            if score_result:
                self._merge_scores(state, score_result["per_item_scores"])
                state.opinion = score_result.get("opinion") or state.opinion
                state.analysis = self._analysis_from_scores(state)

        extra_payload["analysis"] = (
            copy.deepcopy(state.analysis) if state.analysis is not None else None
        )

        try:
            last_clarify = self.repo.get_last_clarify_need(sid)
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to read last clarify target for %s", sid)
            last_clarify = None

        if not clarify_prompt and last_clarify and last_clarify.get("item_id") == item_id:
            try:
                self.repo.clear_last_clarify_need(sid)
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception("Failed to clear clarify target for %s", sid)

        if clarify_prompt and state.clarify < 2:
            try:
                self.repo.set_last_clarify_need(
                    sid, clarify_item_id or item_id, clarify_prompt
                )
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception("Failed to persist clarify target for %s", sid)
            state.clarify += 1
            self._persist_state(state)
            return self._make_response(
                sid,
                state,
                clarify_prompt,
                turn_type="clarify",
                extra=extra_payload,
            )

        state.clarify = 0

        next_item = get_next_item(item_id)
        if next_item != -1:
            self._advance_to(sid, next_item, state)
            next_question = self._generate_primary_question(
                sid, state, next_item, transcripts, dialogue
            )
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
            url = self.tts.synthesize(sid, text)
            if getattr(self.tts, "last_upload", None):
                try:
                    self.repo.save_oss_reference(
                        sid,
                        {
                            "type": "tts",
                            "url": self.tts.last_upload.get("url"),
                            "oss_key": self.tts.last_upload.get("oss_key"),
                            "text": text,
                        },
                    )
                except Exception:  # pragma: no cover - repository guard
                    LOGGER.exception("Failed to persist TTS OSS reference for %s", sid)
            return url
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
        if "risk_hold" in raw:
            setattr(state, "risk_hold", bool(raw.get("risk_hold")))
        elif not hasattr(state, "risk_hold"):
            setattr(state, "risk_hold", False)
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
                "risk_hold": getattr(state, "risk_hold", False),
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

    def _build_dialogue_payload(
        self,
        sid: str,
        transcripts: Optional[List[Dict[str, Any]]] = None,
        *,
        window_turns: int = 3,
    ) -> List[Dict[str, Any]]:
        source = transcripts
        if source is None:
            source = self.repo.get_transcripts(sid) or []

        trimmed: List[Dict[str, Any]] = []
        assistant_turns = 0
        user_turns = 0
        for segment in reversed(source):
            role = segment.get("role")
            if not role:
                role = "assistant" if segment.get("speaker") != "patient" else "user"
            if role == "assistant":
                assistant_turns += 1
            elif role == "user":
                user_turns += 1
            trimmed.append(segment)
            if assistant_turns >= window_turns and user_turns >= window_turns:
                break

        trimmed.reverse()

        dialogue: List[Dict[str, Any]] = []
        for segment in trimmed:
            role = segment.get("role")
            if not role:
                role = "assistant" if segment.get("speaker") != "patient" else "user"
            entry = {
                "sid": sid,
                "utt_id": segment.get("utt_id"),
                "role": role,
                "type": segment.get("type")
                or ("ask" if role == "assistant" else "answer"),
                "text": segment.get("text", ""),
                "ts": segment.get("ts") or [0, 0],
                "sentiment": segment.get("sentiment", "中性"),
            }
            dialogue.append(entry)
        return dialogue

    def _run_ds_analysis_stream(
        self, dialogue: List[Dict[str, Any]]
    ) -> Optional[HAMDResult]:
        if not dialogue or not self.deepseek.usable():
            return None
        started = monotonic()
        try:
            result = self.deepseek.analyze(
                dialogue,
                get_prompt_hamd17(),
                stream=True,
            )
            elapsed = monotonic() - started
            LOGGER.debug("DeepSeek analysis stream completed in %.2fs", elapsed)
            return result
        except DeepSeekTemporarilyUnavailableError as exc:
            LOGGER.debug("DeepSeek analysis temporarily unavailable: %s", exc)
            return None
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("DeepSeek analysis skipped: %s", exc)
            return None
        finally:
            if 'elapsed' not in locals():
                LOGGER.debug("DeepSeek analysis stream aborted after %.2fs", monotonic() - started)

    def _store_analysis_scores(
        self, sid: str, state: SessionState, result: HAMDResult
    ) -> None:
        normalized_items: List[Dict[str, Any]] = []
        for item in result.items:
            normalized = self._normalize_score_entry(
                {
                    "item_id": item.item_id,
                    "score": item.score,
                    "max_score": MAX_SCORE.get(item.item_id, 4),
                    "evidence_refs": item.evidence_refs,
                    "score_type": item.score_type,
                    "score_reason": item.score_reason,
                    "dialogue_evidence": getattr(item, "dialogue_evidence", None),
                    "symptom_summary": getattr(item, "symptom_summary", None),
                    "clarify_need": item.clarify_prompt if item.clarify_need else None,
                }
            )
            if normalized:
                normalized_items.append(normalized)

        if normalized_items:
            self._merge_scores(state, normalized_items)

        analysis_snapshot = (
            self._analysis_from_scores(state) if state.scores_acc else {}
        )
        summary = getattr(result, "summary", None)
        if summary:
            analysis_snapshot = dict(analysis_snapshot or {})
            analysis_snapshot["summary"] = summary
        state.analysis = analysis_snapshot or None

        if state.analysis:
            total_payload = state.analysis.get("total_score") or {}
            total_value = self._extract_total_score(total_payload)
            if total_value is not None:
                state.opinion = state.opinion or self._opinion_from_total(total_value)

        try:
            repository.save_scores(sid, result.model_dump())
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to persist DeepSeek result for %s", sid)
        self._persist_state(state)

orchestrator = LangGraphMini()
