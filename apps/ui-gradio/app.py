from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import ssl
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
import requests
import websockets
import websockets.exceptions

from packages.common.config import settings

from services.audio.tingwu_async_client import (
    create_realtime_task,
    stop_realtime_task,
)
from services.llm.json_client import (
    DeepSeekTemporarilyUnavailableError,
    client as deepseek_client,
)
from services.oss.client import OSSClient

API_BASE = (
    os.getenv("DM_API_BASE", os.getenv("API_BASE_URL", "http://localhost:8080"))
    or "http://localhost:8080"
).rstrip("/")
def _init_session() -> str:
    return str(uuid.uuid4())


def _call_dm_step(
    sid: str, text: Optional[str] = None, audio_ref: Optional[str] = None
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"sid": sid}
    if text:
        payload["text"] = text
    if audio_ref:
        payload["audio_ref"] = audio_ref

    response = requests.post(f"{API_BASE}/dm/step", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _upload_audio(sid: str, file_path: str) -> str:
    url = f"{API_BASE}/upload/audio"
    file_name = Path(file_path).name or "audio.wav"
    with open(file_path, "rb") as handle:
        files = {"file": (file_name, handle, "application/octet-stream")}
        data = {"sid": sid}
        response = requests.post(url, files=files, data=data, timeout=60)
    response.raise_for_status()
    payload = response.json()
    audio_ref = payload.get("audio_ref")
    if not audio_ref:
        raise ValueError("audio_ref missing in upload response")
    return audio_ref


def _generate_report(session_id: str) -> str:
    try:
        resp = requests.post(
            f"{API_BASE}/report/build", json={"sid": session_id}, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        url = data.get("report_url")
        if url:
            return f"✅ 报告已生成：{url}"
        return "⚠️ 报告生成成功但未返回链接。"
    except Exception as exc:  # noqa: BLE001 - surface to UI
        return f"❌ 报告生成失败：{exc}"


def user_step(
    message: str,
    audio_path: Optional[str],
    history: List[Tuple[str, str]],
    session_id: str,
) -> Tuple[List[Tuple[str, str]], str, Dict[str, Any], str, Optional[str]]:
    message = message or ""
    text_payload = message.strip() or None
    audio_ref: Optional[str] = None
    risk_text = "无紧急风险提示。"
    progress: Dict[str, Any] = {}
    audio_value: Optional[str] = None

    try:
        if audio_path:
            audio_ref = _upload_audio(session_id, audio_path)
            if not text_payload:
                text_payload = None

        result = _call_dm_step(session_id, text=text_payload, audio_ref=audio_ref)
    except Exception as exc:  # noqa: BLE001 - surface API failures to the UI
        if text_payload:
            user_label = message
        elif audio_path:
            user_label = f"[音频] {Path(audio_path).name}"
        else:
            user_label = "[空输入]"

        history = history + [(user_label, f"❌ 请求失败：{exc}")]
        return history, "⚠️ 请求失败，请稍后重试。", {}, session_id, None

    assistant_reply = result.get("next_utterance", "")
    previews = result.get("segments_previews") or []
    tts_url = result.get("tts_url")
    if tts_url:
        if tts_url.startswith("file://"):
            local_path = tts_url[7:]
            if Path(local_path).exists():
                audio_value = local_path
        else:
            audio_value = tts_url

    if previews:
        recent_previews = previews[-2:]
        preview_text = "\n".join(f"- {item}" for item in recent_previews if item)
        if preview_text:
            assistant_reply = f"{assistant_reply}\n\n_最近转写预览_:\n{preview_text}"

    user_label: Optional[str] = None
    if text_payload:
        user_label = message
    elif audio_path:
        user_label = f"[音频] {Path(audio_path).name}"

    if user_label:
        history = history + [(user_label, assistant_reply)]
    else:
        history = history + [(None, assistant_reply)]

    progress = result.get("progress", {})
    risk_flag = result.get("risk_flag", False)
    risk_text = (
        "⚠️ 检测到高风险，请立即寻求紧急帮助。" if risk_flag else "无紧急风险提示。"
    )

    return history, risk_text, progress, session_id, audio_value


@dataclass(slots=True)
class TingwuStreamConfig:
    """听悟实时流配置。"""

    appkey: str
    format: str = "pcm"
    language: str = "cn"
    sample_rate: int = 16000
    frame_ms: int = 20


class RealTimeTingwuClient:
    """真正的实时听悟客户端。"""

    def __init__(
        self, complete_sentence_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        self.config = self._build_config()
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.task_id: Optional[str] = None
        self.is_connected = False
        self.audio_queue: "queue.Queue[tuple[int, np.ndarray]]" = queue.Queue()
        self.result_queue: "queue.Queue[str]" = queue.Queue()
        self.complete_sentence_queue: "queue.Queue[str]" = queue.Queue()
        self.running = False
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        self.current_sentence = ""
        self.last_audio_time = 0.0
        self.silence_packet: Optional[bytes] = None
        self.keepalive_task: Optional[asyncio.Task[None]] = None
        self.complete_sentence_callback = complete_sentence_callback

    def _build_config(self) -> TingwuStreamConfig:
        appkey = settings.TINGWU_APPKEY or settings.ALIBABA_TINGWU_APPKEY
        if not appkey:
            raise RuntimeError("请在环境变量或 .env 中配置 TINGWU_APPKEY")
        return TingwuStreamConfig(
            appkey=appkey,
            format=settings.TINGWU_FORMAT or "pcm",
            language=settings.TINGWU_LANG or "cn",
            sample_rate=16000,
        )

    def _ensure_mono(self, audio_data: np.ndarray) -> np.ndarray:
        if audio_data.ndim == 1:
            return audio_data
        return np.mean(audio_data, axis=1)

    def _resample_audio(
        self, audio_data: np.ndarray, original_sr: int, target_sr: int
    ) -> np.ndarray:
        if original_sr == target_sr:
            return audio_data
        if audio_data.size == 0:
            return audio_data
        duration = len(audio_data) / original_sr
        target_length = int(duration * target_sr)
        if target_length == 0:
            return np.array([], dtype=np.float32)
        original_indices = np.arange(len(audio_data), dtype=np.float32)
        target_indices = np.linspace(0, len(audio_data) - 1, target_length, dtype=np.float32)
        return np.interp(target_indices, original_indices, audio_data).astype(np.float32)

    def _audio_to_pcm(self, audio_data: np.ndarray) -> bytes:
        normalised = np.clip(audio_data, -1.0, 1.0)
        return (normalised * 32767).astype("<i2").tobytes()

    def _generate_silence_packet(self, duration_ms: int = 100) -> bytes:
        samples = int(self.config.sample_rate * duration_ms / 1000)
        silence_data = np.zeros(samples, dtype=np.float32)
        return self._audio_to_pcm(silence_data)

    async def connect(self) -> bool:
        self.connection_attempts += 1
        try:
            ws_url, self.task_id = await create_realtime_task()
            print(f"🎯 创建听悟任务: {self.task_id}")
            connect_kwargs: Dict[str, Any] = {
                "ping_interval": 10,
                "ping_timeout": 30,
                "max_size": 10 * 1024 * 1024,
            }
            try:
                self.ws = await websockets.connect(ws_url, **connect_kwargs)
            except Exception as exc:
                if "SSL" in str(exc) or "302" in str(exc):
                    connect_kwargs["ssl"] = ssl._create_unverified_context()
                    self.ws = await websockets.connect(ws_url, **connect_kwargs)
                else:
                    raise
            self.is_connected = True
            self.connection_attempts = 0
            self.last_audio_time = time.time()
            self.silence_packet = self._generate_silence_packet(100)
            print("✅ WebSocket连接成功")
            start_message = {
                "header": {
                    "namespace": "SpeechTranscriber",
                    "name": "StartTranscription",
                    "task_id": self.task_id,
                },
                "payload": {
                    "format": self.config.format,
                    "sample_rate": self.config.sample_rate,
                    "enable_chinese_english_translation": False,
                    "enable_speech_detection": True,
                    "enable_words": False,
                    "enable_intermediate_result": True,
                    "enable_punctuation_prediction": True,
                    "enable_inverse_text_normalization": True,
                    "enable_semantic_sentence_detection": True,
                    "max_sentence_silence": 800,
                },
            }
            await self.ws.send(json.dumps(start_message, ensure_ascii=False))
            print("🚀 开始实时转录")
            return True
        except Exception as exc:
            print(
                f"❌ 连接失败 (尝试 {self.connection_attempts}/{self.max_connection_attempts}): {exc}"
            )
            self.result_queue.put(f"❌ 连接失败: {exc}")
            if self.connection_attempts >= self.max_connection_attempts:
                self.result_queue.put("❌ 连接失败次数过多，请检查网络和配置")
            return False

    async def send_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int) -> None:
        if not self.is_connected or not self.ws:
            return
        try:
            mono_audio = self._ensure_mono(audio_chunk.astype(np.float32))
            processed_audio = self._resample_audio(
                mono_audio, sample_rate, self.config.sample_rate
            )
            if processed_audio.size == 0:
                return
            pcm_data = self._audio_to_pcm(processed_audio)
            max_chunk_size = 16 * 1024
            if len(pcm_data) > max_chunk_size:
                for index in range(0, len(pcm_data), max_chunk_size):
                    chunk = pcm_data[index : index + max_chunk_size]
                    await self.ws.send(chunk)
                    self.last_audio_time = time.time()
                    await asyncio.sleep(0.001)
            else:
                await self.ws.send(pcm_data)
                self.last_audio_time = time.time()
        except Exception as exc:
            print(f"⚠️ 发送音频失败: {exc}")

    async def keepalive_loop(self) -> None:
        while self.running and self.is_connected:
            try:
                current_time = time.time()
                if current_time - self.last_audio_time > 3:
                    await self.send_keepalive_silence()
                    self.last_audio_time = current_time
                await asyncio.sleep(1)
            except Exception as exc:
                print(f"⚠️ 保持连接循环错误: {exc}")
                break

    async def send_keepalive_silence(self) -> None:
        if not self.is_connected or not self.ws or not self.silence_packet:
            return
        try:
            await self.ws.send(self.silence_packet)
            print("🔇 发送静音包保持连接")
        except Exception as exc:
            print(f"⚠️ 发送静音包失败: {exc}")

    async def receive_messages(self) -> None:
        if not self.is_connected or not self.ws:
            return
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as exc:
                    print(f"⚠️ JSON解析错误: {exc}")
                    continue
                except Exception as exc:
                    print(f"⚠️ 消息处理错误: {exc}")
                    continue
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed as exc:
            if self.running:
                print(f"⚠️ WebSocket连接关闭: {exc}")
                self.result_queue.put(f"⚠️ 连接中断: {exc}")
        except Exception as exc:
            if self.running:
                print(f"⚠️ 接收消息错误: {exc}")
                self.result_queue.put(f"⚠️ 连接错误: {exc}")

    def _extract_text_from_payload(self, payload: dict) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        text_fields = ["result", "text", "transcript", "asr_text"]
        for field in text_fields:
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        words = payload.get("words")
        if isinstance(words, list):
            word_texts = [
                word.get("text")
                for word in words
                if isinstance(word, dict) and isinstance(word.get("text"), str)
            ]
            cleaned = " ".join(text for text in word_texts if text)
            if cleaned.strip():
                return cleaned.strip()
        sentences = payload.get("sentences")
        if isinstance(sentences, list) and sentences:
            collected = []
            for sentence in sentences:
                if isinstance(sentence, dict):
                    text = sentence.get("text")
                    if isinstance(text, str) and text.strip():
                        collected.append(text.strip())
            if collected:
                return " ".join(collected)
        return None

    async def _handle_message(self, data: dict) -> None:
        header = data.get("header", {})
        payload = data.get("payload", {})
        name = header.get("name")
        status = header.get("status")
        print(f"📥 收到消息: {name}")
        if isinstance(status, int) and status >= 40_000_000:
            error_msg = payload.get("message", payload.get("error_message", "未知错误"))
            detailed_error = f"❌ 听悟错误({status}): {error_msg}"
            print(detailed_error)
            self.result_queue.put(detailed_error)
            self.running = False
            return
        if name == "TranscriptionStarted":
            self.result_queue.put("🎤 实时转录已开始，请说话...")
        elif name == "SentenceBegin":
            self.current_sentence = ""
            self.result_queue.put("🔊 检测到语音开始")
        elif name in {"TranscriptionResult", "TranscriptionResultChanged"}:
            result_text = self._extract_text_from_payload(payload)
            if result_text:
                self.current_sentence = result_text
                display_result = result_text
                if len(display_result) > 500:
                    display_result = display_result[:500] + "..."
                prefix = "📝" if name == "TranscriptionResult" else "🔄"
                self.result_queue.put(f"{prefix} {display_result}")
            else:
                print(f"📥 {name}: 无文本内容")
        elif name == "SentenceEnd":
            result_text = self._extract_text_from_payload(payload)
            if not result_text and self.current_sentence:
                result_text = self.current_sentence
            if result_text:
                self.complete_sentence_queue.put(result_text)
                if self.complete_sentence_callback:
                    self.complete_sentence_callback(result_text)
                print(f"🎯 [完整句子] {result_text}")
                self.result_queue.put(f"✅ {result_text}")
                self.current_sentence = ""
            else:
                self.result_queue.put("⏹️ 句子结束")
        elif name == "TranscriptionCompleted":
            self.result_queue.put("🏁 转录完成")
        elif name == "TaskFailed":
            error_msg = payload.get("message", payload.get("error_message", "任务执行失败"))
            detailed_error = f"❌ 任务失败: {error_msg}"
            print(detailed_error)
            self.result_queue.put(detailed_error)
            self.running = False
        elif name == "SpeechStartDetected":
            self.result_queue.put("🔊 检测到语音开始")
        elif name == "SpeechEndDetected":
            self.result_queue.put("🔇 检测到语音结束")
        else:
            print(
                f"📥 未知消息类型: {name}, payload: {json.dumps(payload, ensure_ascii=False)[:200]}"
            )

    async def start_realtime_stream(self) -> None:
        self.running = True
        if not await self.connect():
            self.running = False
            return
        receive_task = asyncio.create_task(self.receive_messages())
        self.keepalive_task = asyncio.create_task(self.keepalive_loop())
        try:
            while self.running:
                try:
                    sample_rate, audio_chunk = self.audio_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                try:
                    if audio_chunk.size > 0:
                        await self.send_audio_chunk(audio_chunk, sample_rate)
                except Exception as exc:
                    print(f"⚠️ 处理音频数据错误: {exc}")
                    await asyncio.sleep(0.01)
        except Exception as exc:
            print(f"❌ 实时流错误: {exc}")
            self.result_queue.put(f"❌ 实时流错误: {exc}")
        finally:
            self.running = False
            if not receive_task.done():
                receive_task.cancel()
            if self.keepalive_task and not self.keepalive_task.done():
                self.keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receive_task
                if self.keepalive_task:
                    await self.keepalive_task
            await self.close()

    def add_audio_data(self, sample_rate: int, audio_data: np.ndarray) -> None:
        if self.running and audio_data is not None and audio_data.size > 0:
            self.audio_queue.put((sample_rate, audio_data))

    def get_results(self) -> List[str]:
        results: List[str] = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def get_complete_sentences(self) -> List[str]:
        sentences: List[str] = []
        while not self.complete_sentence_queue.empty():
            try:
                sentences.append(self.complete_sentence_queue.get_nowait())
            except queue.Empty:
                break
        return sentences

    async def close(self) -> None:
        self.running = False
        if self.is_connected and self.ws:
            try:
                stop_message = {
                    "header": {
                        "namespace": "SpeechTranscriber",
                        "name": "StopTranscription",
                    },
                    "payload": {},
                }
                await self.ws.send(json.dumps(stop_message, ensure_ascii=False))
                await asyncio.sleep(1)
            except Exception as exc:
                print(f"⚠️ 发送停止消息时出错: {exc}")
            finally:
                try:
                    await self.ws.close()
                except Exception as exc:
                    print(f"⚠️ 关闭WebSocket时出错: {exc}")
                self.is_connected = False
                print("🔚 WebSocket连接已关闭")
        if self.task_id:
            try:
                await stop_realtime_task(self.task_id)
                print("🔚 听悟任务已停止")
            except Exception as exc:
                print(f"⚠️ 停止任务失败: {exc}")


_client: Optional[RealTimeTingwuClient] = None
_client_lock = threading.Lock()
_complete_sentences: List[str] = []
_oss_client = OSSClient()


def handle_complete_sentence(sentence: str) -> None:
    """接收听悟返回的完整句子（原始文本）。"""

    print(f"🎯 [问答流程] 收到完整句子: {sentence}")

        self.connection_attempts += 1

def get_latest_complete_sentence() -> Optional[str]:
    if _complete_sentences:
        return _complete_sentences[-1]
    return None


def get_all_complete_sentences() -> List[str]:
    return _complete_sentences.copy()


def clear_complete_sentences() -> None:
    _complete_sentences.clear()


def _format_sentences_display(sentences: List[str]) -> str:
    if not sentences:
        return "暂无完整句子"
    return "\n\n".join(f"{index + 1}. {value}" for index, value in enumerate(sentences))


def _ensure_audio_playable_url(session_id: str, audio_value: Optional[str]) -> Optional[str]:
    """将本地音频转换为 OSS 公网链接，便于前端播放。"""

    if not audio_value:
        return None
    if isinstance(audio_value, str) and audio_value.startswith(("http://", "https://")):
        return audio_value
    local_path = audio_value
    if isinstance(local_path, str) and local_path.startswith("file://"):
        local_path = local_path[7:]
    if not local_path or not Path(local_path).exists():
        return audio_value
    if _oss_client.enabled:
        try:
            url = _oss_client.store_artifact(session_id, "tts", local_path)
            if url:
                return url
        except Exception as exc:
            print(f"⚠️ 上传音频到OSS失败: {exc}")
    return local_path


def _clean_sentence_sync(sentence: str) -> str:
    """调用 DeepSeek 对转写文本进行清洗（同步执行）。"""

    text = sentence.strip()
    if not text:
        return ""
    if not deepseek_client.enabled():
        return text
    messages = [
        {
            "role": "system",
            "content": "你是语音识别文本的清洗助手，请保留语义并修正口语化错误，输出纯文本。",
        },
        {"role": "user", "content": text},
    ]
    try:
        cleaned = deepseek_client._post_chat(  # type: ignore[attr-defined]
            messages=messages,
            max_tokens=512,
            temperature=0.2,
        )
        cleaned_text = cleaned.strip()
        return cleaned_text or text
    except DeepSeekTemporarilyUnavailableError:
        return text
    except Exception as exc:
        print(f"⚠️ DeepSeek 清洗失败: {exc}")
        return text


async def _clean_sentence(sentence: str) -> str:
    return await asyncio.to_thread(_clean_sentence_sync, sentence)


async def start_realtime_transcription(
    audio: Optional[Tuple[int, np.ndarray]]
) -> AsyncGenerator[str, None]:
    global _client
    try:
        if audio is None:
            yield "请开始说话..."
            return
        sample_rate, audio_data = audio
        if audio_data is None or audio_data.size == 0:
            yield "未检测到有效音频数据"
            return
        with _client_lock:
            if _client is None or not _client.running:
                _client = RealTimeTingwuClient(
                    complete_sentence_callback=handle_complete_sentence
                )
                asyncio.create_task(_client.start_realtime_stream())
                await asyncio.sleep(2)
        _client.add_audio_data(sample_rate, audio_data)
        results = _client.get_results()
        for result in results:
            yield result
        await asyncio.sleep(0.1)
    except Exception as exc:
        print(f"❌ 实时转录错误: {exc}")
        yield f"❌ 系统错误: {exc}"


async def stop_transcription() -> str:
    global _client
    with _client_lock:
        client = _client
        _client = None
    if client is not None:
        await client.close()
    return "🛑 转录已停止"


async def realtime_conversation(
    audio: Optional[Tuple[int, np.ndarray]],
    history: Optional[List[Tuple[str, str]]],
    session_id: str,
    risk_text: str,
    progress: Optional[Dict[str, Any]],
    audio_value: Optional[str],
    existing_log: str,
    latest_sentence_text: str,
    all_sentences_text: str,
) -> AsyncGenerator[
    Tuple[
        str,
        List[Tuple[str, str]],
        str,
        Dict[str, Any],
        Optional[str],
        str,
        str,
        str,
    ],
    None,
]:
    """实时语音处理与问答联动生成器。"""

    log_lines: List[str] = []
    if existing_log:
        log_lines.extend(existing_log.splitlines())
    current_history: List[Tuple[str, str]] = history or []
    current_session = session_id or _init_session()
    current_risk = risk_text or "无紧急风险提示。"
    current_progress: Dict[str, Any] = progress or {}
    current_audio = audio_value
    latest_sentence_cached = latest_sentence_text or "暂无完整句子"
    all_sentences_cached = all_sentences_text or "暂无完整句子"

    async for message in start_realtime_transcription(audio):
        if message:
            log_lines.append(message)
        pending_sentences: List[str] = []
        with _client_lock:
            if _client is not None:
                pending_sentences = _client.get_complete_sentences()
        for raw_sentence in pending_sentences:
            cleaned_sentence = await _clean_sentence(raw_sentence)
            if not cleaned_sentence:
                continue
            _complete_sentences.append(cleaned_sentence)
            latest_sentence_cached = cleaned_sentence
            all_sentences_cached = _format_sentences_display(_complete_sentences)
            log_lines.append(f"🤖 清洗后进入问答：{cleaned_sentence}")
            try:
                (
                    current_history,
                    current_risk,
                    current_progress,
                    current_session,
                    step_audio,
                ) = await asyncio.to_thread(
                    user_step,
                    cleaned_sentence,
                    None,
                    current_history,
                    current_session,
                )
                current_audio = _ensure_audio_playable_url(current_session, step_audio)
            except Exception as exc:
                error_text = f"❌ 问答处理失败：{exc}"
                log_lines.append(error_text)
        yield (
            "\n".join(log_lines[-200:]),
            current_history,
            current_risk,
            current_progress,
            current_audio,
            current_session,
            latest_sentence_cached,
            all_sentences_cached,
        )


def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="抑郁随访助手") as demo:
        session_state = gr.State(_init_session())

        gr.Markdown(
            """
            # 抑郁随访助手
            集成文本/音频问答、实时语音识别与 DeepSeek 清洗，完成自动随访与报告。
            """
        )

        with gr.Tabs():
            with gr.Tab("评估"):
                chatbot = gr.Chatbot(height=420, label="对话记录")

                with gr.Row():
                    with gr.Column(scale=3):
                        risk_alert = gr.Markdown("无紧急风险提示。")
                        progress_display = gr.JSON(label="进度状态")
                        audio_sys = gr.Audio(
                            label="系统语音播放", interactive=False, autoplay=True
                        )
                        text_input = gr.Textbox(
                            label="患者文本输入", placeholder="请输入文本信息"
                        )
                        audio_input = gr.File(
                            label="上传音频 (16kHz 单声道)", type="filepath"
                        )
                        send_button = gr.Button("发送文本/音频", variant="primary")
                        gr.Markdown(
                            "提示：可手动输入文本或上传录音文件，系统会同步更新问答。"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 🎧 实时语音识别")
                        mic = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            streaming=True,
                            label="点击麦克风即可推流至听悟",
                        )
                        stop_btn = gr.Button("🛑 停止转录", variant="stop")
                        realtime_output = gr.Textbox(
                            label="实时识别日志",
                            lines=16,
                            max_lines=20,
                            show_copy_button=True,
                            interactive=False,
                        )
                        with gr.Accordion("📝 完整句子（已清洗，用于问答）", open=False):
                            latest_sentence = gr.Textbox(
                                label="最新完整句子",
                                lines=2,
                                interactive=False,
                                placeholder="暂无完整句子",
                            )
                            all_sentences = gr.Textbox(
                                label="全部完整句子",
                                lines=6,
                                interactive=False,
                                placeholder="暂无完整句子",
                            )
                            with gr.Row():
                                refresh_btn = gr.Button("🔄 刷新句子列表", size="sm")
                                clear_btn = gr.Button("🗑️ 清空句子列表", size="sm")
                        gr.Markdown(
                            """
                            - 实时语音会自动送入 DeepSeek 清洗，并进入问答流程。
                            - ✅ 提示代表听悟返回的完整句子。
                            - 清洗失败或网络异常时会在日志中提示。
                            """
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 控制面板")
                        mic = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            streaming=True,
                            label="🎙️ 麦克风输入 (16kHz 单声道)",
                            show_download_button=False,
                        )
                        stop_btn = gr.Button("🛑 停止转录", variant="stop", size="lg")

                        with gr.Accordion("📝 完整句子（问答流程依据）", open=False):
                            latest_sentence = gr.Textbox(
                                label="最新完整句子",
                                lines=2,
                                placeholder="这里将显示最新的完整句子...",
                                interactive=False,
                            )
                            all_sentences = gr.Textbox(
                                label="所有完整句子",
                                lines=5,
                                placeholder="这里将显示全部完整句子...",
                                interactive=False,
                            )
                            with gr.Row():
                                refresh_btn = gr.Button("🔄 刷新句子列表", size="sm")
                                clear_btn = gr.Button("🗑️ 清空句子列表", size="sm")

                        gr.Markdown(
                            """
                            **提示：**
                            - 建议在安静环境下发言，保持语速适中
                            - 系统会自动发送静音包维持连接
                            - DeepSeek 会自动清洗识别句子
                            - 清洗后的句子已同步至评估问答流程
                            """
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 实时字幕")
                        realtime_output = gr.Textbox(
                            label="识别结果",
                            lines=15,
                            max_lines=20,
                            show_copy_button=True,
                            autoscroll=True,
                            placeholder="识别结果将实时显示在这里...",
                            elem_id="realtime_output",
                        )

            with gr.Tab("报告"):
                gr.Markdown("## 生成评估报告")
                gr.Markdown("点击按钮后将在 /tmp/depression_agent_reports/ 下生成 PDF。")
                report_button = gr.Button("生成报告")
                report_status = gr.Markdown("等待生成指令…")

        def _on_submit(
            message: str,
            audio_path: Optional[str],
            history: List[Tuple[str, str]],
            session_id: str,
        ) -> Tuple[List[Tuple[str, str]], str, Optional[str], str, str, Dict[str, Any], Optional[Any]]:
            chat, risk_text, progress, sid, audio_value = user_step(
                message, audio_path, history, session_id
            )
            playable_audio = _ensure_audio_playable_url(sid, audio_value)
            return chat, "", None, sid, risk_text, progress, playable_audio

        text_input.submit(
            _on_submit,
            inputs=[text_input, audio_input, chatbot, session_state],
            outputs=[
                chatbot,
                text_input,
                audio_input,
                session_state,
                risk_alert,
                progress_display,
                audio_sys,
            ],
        )

        send_button.click(
            _on_submit,
            inputs=[text_input, audio_input, chatbot, session_state],
            outputs=[
                chatbot,
                text_input,
                audio_input,
                session_state,
                risk_alert,
                progress_display,
                audio_sys,
            ],
        )

        mic.stream(
            fn=realtime_conversation,
            inputs=[
                mic,
                chatbot,
                session_state,
                risk_alert,
                progress_display,
                audio_sys,
                realtime_output,
                latest_sentence,
                all_sentences,
            ],
            outputs=[
                realtime_output,
                chatbot,
                risk_alert,
                progress_display,
                audio_sys,
                session_state,
                latest_sentence,
                all_sentences,
            ],
            show_progress="hidden",
        )

        stop_btn.click(
            fn=stop_transcription,
            outputs=[realtime_output],
        )

        def refresh_sentences() -> Tuple[str, str]:
            """刷新清洗后句子的展示。"""

            sentences = get_all_complete_sentences()
            latest = get_latest_complete_sentence() or "暂无完整句子"
            return latest, _format_sentences_display(sentences)

        refresh_btn.click(
            fn=refresh_sentences,
            outputs=[latest_sentence, all_sentences],
        )

        def clear_sentences_action() -> Tuple[str, str]:
            """清空完整句子列表。"""

            clear_complete_sentences()
            return "已清空", "已清空"

        clear_btn.click(
            fn=clear_sentences_action,
            outputs=[latest_sentence, all_sentences],
        )

        report_button.click(
            lambda sid: _generate_report(sid),
            inputs=[session_state],
            outputs=[report_status],
        )

    return demo


if __name__ == "__main__":
    print("🚀 启动听悟实时语音识别应用...")
    print(f"📁 项目根目录: {PROJECT_ROOT}")
    print(
        "🔑 听悟 AppKey: ",
        settings.TINGWU_APPKEY or settings.ALIBABA_TINGWU_APPKEY or "未配置",
    )
    build_ui().launch(server_name="0.0.0.0", server_port=8001)
