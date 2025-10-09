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
    """听悟实时流配置"""

    appkey: str
    format: str = "pcm"
    language: str = "cn"
    sample_rate: int = 16000
    frame_ms: int = 20  # 20ms帧大小


class RealTimeTingwuClient:
    """真正的实时听悟客户端 - 修复连接问题"""

    def __init__(self, complete_sentence_callback: Optional[Callable[[str], None]] = None):
        self.config = self._build_config()
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.task_id: Optional[str] = None
        self.is_connected = False
        self.audio_queue: "queue.Queue[Tuple[int, np.ndarray]]" = queue.Queue()
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
            raise RuntimeError("请在 .env 中配置 TINGWU_APPKEY")

        return TingwuStreamConfig(
            appkey=appkey,
            format=settings.TINGWU_FORMAT or "pcm",
            language=settings.TINGWU_LANG or "cn",
            sample_rate=16000,
        )

    def _ensure_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """确保音频为单声道"""

        if audio_data.ndim == 1:
            return audio_data
        return np.mean(audio_data, axis=1)

    def _resample_audio(
        self, audio_data: np.ndarray, original_sr: int, target_sr: int
    ) -> np.ndarray:
        """重采样音频到目标采样率"""

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
        """将numpy数组转换为PCM字节"""

        normalized = np.clip(audio_data, -1.0, 1.0)
        return (normalized * 32767).astype("<i2").tobytes()

    def _generate_silence_packet(self, duration_ms: int = 100) -> bytes:
        """生成静音数据包，用于保持连接活跃"""

        samples = int(self.config.sample_rate * duration_ms / 1000)
        silence_data = np.zeros(samples, dtype=np.float32)
        return self._audio_to_pcm(silence_data)

    async def connect(self) -> bool:
        """连接到听悟WebSocket"""

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
            return False

    async def send_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int) -> None:
        """发送音频数据块"""

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
                for offset in range(0, len(pcm_data), max_chunk_size):
                    chunk = pcm_data[offset : offset + max_chunk_size]
                    await self.ws.send(chunk)
                    self.last_audio_time = time.time()
                    await asyncio.sleep(0.001)
            else:
                await self.ws.send(pcm_data)
                self.last_audio_time = time.time()

        except Exception as exc:
            print(f"⚠️ 发送音频失败: {exc}")

    async def keepalive_loop(self) -> None:
        """保持连接活跃的循环任务"""

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
        """发送静音包保持连接活跃"""

        if not self.is_connected or not self.ws or not self.silence_packet:
            return

        try:
            await self.ws.send(self.silence_packet)
            print("🔇 发送静音包保持连接")
        except Exception as exc:
            print(f"⚠️ 发送静音包失败: {exc}")

    async def receive_messages(self) -> None:
        """接收WebSocket消息"""

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
        """从payload中提取文本内容 - 根据官方文档格式"""

        if not isinstance(payload, dict):
            return None

        text_fields = ["result", "text", "transcript", "asr_text"]
        for field in text_fields:
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()

        words = payload.get("words")
        if isinstance(words, list):
            texts = []
            for word in words:
                if isinstance(word, dict) and word.get("text"):
                    texts.append(str(word.get("text")))
            if texts:
                return " ".join(texts)

        sentences = payload.get("sentences")
        if isinstance(sentences, list) and sentences:
            texts = []
            for sentence in sentences:
                if isinstance(sentence, dict):
                    text = sentence.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
            if texts:
                return " ".join(texts)

        return None

    async def _handle_message(self, data: dict) -> None:
        """处理WebSocket消息 - 根据官方文档的事件类型"""

        header = data.get("header", {})
        payload = data.get("payload", {})
        name = header.get("name")
        status = header.get("status")

        print(f"📥 收到消息: {name}")

        if isinstance(status, int) and status >= 40000000:
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

        elif name in ["TranscriptionResult", "TranscriptionResultChanged"]:
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
        """开始实时流式处理"""

        self.running = True

        if not await self.connect():
            self.running = False
            return

        receive_task = asyncio.create_task(self.receive_messages())
        self.keepalive_task = asyncio.create_task(self.keepalive_loop())

        try:
            while self.running:
                try:
                    audio_data = self.audio_queue.get_nowait()
                    sample_rate, audio_chunk = audio_data

                    if audio_chunk.size > 0:
                        await self.send_audio_chunk(audio_chunk, sample_rate)

                except queue.Empty:
                    await asyncio.sleep(0.01)
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
        """添加音频数据到处理队列"""

        if self.running and audio_data is not None and audio_data.size > 0:
            self.audio_queue.put((sample_rate, audio_data))

    def get_results(self) -> List[str]:
        """获取识别结果"""

        results: List[str] = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def get_complete_sentences(self) -> List[str]:
        """获取完整的句子列表"""

        sentences: List[str] = []
        while not self.complete_sentence_queue.empty():
            try:
                sentences.append(self.complete_sentence_queue.get_nowait())
            except queue.Empty:
                break
        return sentences

    async def close(self) -> None:
        """关闭连接"""

        self.running = False

        if self.is_connected and self.ws:
            try:
                stop_message = {
                    "header": {"namespace": "SpeechTranscriber", "name": "StopTranscription"},
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
_qa_lock: Optional[asyncio.Lock] = None
_realtime_logs: List[str] = []


def _get_qa_lock() -> asyncio.Lock:
    """懒加载问答锁，保证转写触发的问答串行执行"""

    global _qa_lock
    if _qa_lock is None:
        _qa_lock = asyncio.Lock()
    return _qa_lock


def handle_complete_sentence(sentence: str) -> None:
    """处理完整句子的回调函数"""

    global _complete_sentences
    _complete_sentences.append(sentence)
    print(f"🎯 [问答流程] 收到完整句子: {sentence}")


def get_latest_complete_sentence() -> Optional[str]:
    """获取最新的完整句子 - 用于问答流程"""

    if _complete_sentences:
        return _complete_sentences[-1]
    return None


def get_all_complete_sentences() -> List[str]:
    """获取所有完整句子"""

    return _complete_sentences.copy()


def clear_complete_sentences() -> None:
    """清空完整句子列表"""

    _complete_sentences.clear()


def _format_audio_output(audio_value: Optional[Any]) -> Optional[Any]:
    """统一格式化语音播放值，支持本地文件与公网 URL"""

    if not audio_value:
        return None
    if isinstance(audio_value, dict):
        return audio_value
    if isinstance(audio_value, str):
        if audio_value.startswith(("http://", "https://")):
            return {"url": audio_value}
        return audio_value
    return None


async def clean_sentence_with_deepseek(sentence: str) -> Tuple[str, str]:
    """调用 DeepSeek 对句子进行清洗，返回清洗结果与日志"""

    text = (sentence or "").strip()
    if not text:
        return "", "⚠️ 识别到的句子为空，跳过清洗"

    base = (settings.DEEPSEEK_API_BASE or settings.deepseek_api_base or "").strip()
    key = (settings.DEEPSEEK_API_KEY or settings.deepseek_api_key or "").strip()
    if not base or not key:
        return text, f"⚠️ DeepSeek 未配置，使用原句：{text}"

    def _request() -> str:
        url_root = base.rstrip("/")
        if not url_root.endswith("/v1"):
            url_root = f"{url_root}/v1"
        url = f"{url_root}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        system_prompt = (
            "你是中文语音转写清洗助手。请在不改变原意的前提下，删除口头禅、重复和语气词，"
            "适度补齐省略的主语或宾语，补全标点并输出简体中文。只返回清洗后的文本。"
        )
        payload = {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0.2,
            "max_tokens": 128,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        content = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return content

    try:
        cleaned = await asyncio.to_thread(_request)
    except Exception as exc:  # noqa: BLE001 - 直接反馈清洗异常
        return text, f"⚠️ DeepSeek 清洗失败：{exc}"

    if not cleaned:
        return text, f"⚠️ DeepSeek 清洗返回空结果，已使用原句：{text}"

    if cleaned == text:
        return cleaned, f"🧹 清洗完成：{cleaned}（未做调整）"

    return cleaned, f"🧹 清洗完成：{cleaned}"


async def start_realtime_transcription(
    audio: Optional[Tuple[int, np.ndarray]],
    history: Optional[List[Tuple[str, str]]],
    session_id: Optional[str],
    risk_text: Optional[str],
    progress: Optional[Dict[str, Any]],
    audio_value: Optional[Any],
) -> AsyncGenerator[Tuple[str, List[Tuple[str, str]], str, str, Dict[str, Any], Optional[Any]], None]:
    """开始实时语音识别并触发问答流程"""

    global _client, _realtime_logs

    history = history or []
    session_id = session_id or _init_session()
    risk_text = risk_text or "无紧急风险提示。"
    progress = progress or {}
    formatted_audio = _format_audio_output(audio_value if isinstance(audio_value, str) else audio_value)

    if audio is None:
        log_text = "\n".join(_realtime_logs) if _realtime_logs else "请开始说话..."
        yield log_text, history, session_id, risk_text, progress, formatted_audio
        return

    sample_rate, audio_data = audio
    if audio_data is None or getattr(audio_data, "size", 0) == 0:
        _realtime_logs.append("⚠️ 未检测到有效音频数据")
        yield "\n".join(_realtime_logs[-200:]), history, session_id, risk_text, progress, formatted_audio
        return

    start_client = False
    with _client_lock:
        if _client is None or not _client.running:
            _client = RealTimeTingwuClient(complete_sentence_callback=handle_complete_sentence)
            start_client = True

    if start_client and _client:
        clear_complete_sentences()
        _realtime_logs = ["🎯 正在建立实时转录会话..."]
        asyncio.create_task(_client.start_realtime_stream())
        await asyncio.sleep(2)
        _realtime_logs.append("🎤 实时客户端已就绪，请继续说话")
        yield "\n".join(_realtime_logs[-200:]), history, session_id, risk_text, progress, formatted_audio

    if not _client or not _client.running:
        _realtime_logs.append("❌ 实时客户端未就绪，请稍后重试")
        yield "\n".join(_realtime_logs[-200:]), history, session_id, risk_text, progress, formatted_audio
        return

    _client.add_audio_data(sample_rate, audio_data)

    results = _client.get_results()
    if results:
        _realtime_logs.extend(results)
        yield "\n".join(_realtime_logs[-200:]), history, session_id, risk_text, progress, formatted_audio

    sentences = _client.get_complete_sentences()
    for raw_sentence in sentences:
        cleaned_sentence, clean_log = await clean_sentence_with_deepseek(raw_sentence)
        if clean_log:
            _realtime_logs.append(clean_log)

        if not cleaned_sentence:
            yield "\n".join(_realtime_logs[-200:]), history, session_id, risk_text, progress, formatted_audio
            continue

        async with _get_qa_lock():
            history, risk_text, progress, session_id, audio_value = await asyncio.to_thread(
                user_step,
                cleaned_sentence,
                None,
                history,
                session_id,
            )

        formatted_audio = _format_audio_output(audio_value)
        if history:
            last_reply = history[-1][1]
            if last_reply:
                _realtime_logs.append(f"🤖 助理回复：{last_reply}")

        yield "\n".join(_realtime_logs[-200:]), history, session_id, risk_text, progress, formatted_audio


async def stop_transcription() -> AsyncGenerator[str, None]:
    """停止转录"""

    global _client, _realtime_logs

    with _client_lock:
        current = _client
        _client = None

    if current is not None:
        await current.close()

    clear_complete_sentences()
    _realtime_logs = []

    yield "🛑 转录已停止"


def refresh_sentences() -> Tuple[str, str]:
    """刷新完整句子列表"""

    sentences = get_all_complete_sentences()
    latest = get_latest_complete_sentence() or "暂无完整句子"
    all_text = (
        "\n\n".join([f"{index + 1}. {text}" for index, text in enumerate(sentences)])
        if sentences
        else "暂无完整句子"
    )
    return latest, all_text


def clear_sentences() -> Tuple[str, str]:
    """清空完整句子并返回提示"""

    clear_complete_sentences()
    return "已清空", "已清空"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Depression Agent UI") as demo:
        session_state = gr.State(_init_session())

        gr.Markdown("# 抑郁随访助手")

        with gr.Tabs():
            with gr.Tab("评估"):
                chatbot = gr.Chatbot(height=400, label="对话")
                text_input = gr.Textbox(label="患者输入", placeholder="请输入文本")
                audio_input = gr.File(label="上传音频(16k mono)", type="filepath")
                audio_sys = gr.Audio(label="系统语音", interactive=False, autoplay=True)
                risk_alert = gr.Markdown("无紧急风险提示。")
                progress_display = gr.JSON(label="进度状态")
                send_button = gr.Button("发送")

            with gr.Tab("实时识别"):
                gr.Markdown(
                    """
                    ## 🎧 实时语音识别
                    **基于阿里云听悟实时API的真正实时流式识别**

                    ### 使用说明：
                    1. 点击麦克风开始录音
                    2. 实时识别结果将在右侧展示
                    3. 每条清洗后的句子会自动进入问答环节
                    4. 点击“停止转录”可主动结束本轮会话
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
            formatted_audio = _format_audio_output(audio_value)
            return chat, "", None, sid, risk_text, progress, formatted_audio

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
            fn=start_realtime_transcription,
            inputs=[mic, chatbot, session_state, risk_alert, progress_display, audio_sys],
            outputs=[realtime_output, chatbot, session_state, risk_alert, progress_display, audio_sys],
            show_progress="hidden",
        )

        stop_btn.click(
            fn=stop_transcription,
            outputs=realtime_output,
        )

        refresh_btn.click(
            fn=refresh_sentences,
            outputs=[latest_sentence, all_sentences],
        )

        clear_btn.click(
            fn=clear_sentences,
            outputs=[latest_sentence, all_sentences],
        )

        report_button.click(
            lambda sid: _generate_report(sid),
            inputs=[session_state],
            outputs=[report_status],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
