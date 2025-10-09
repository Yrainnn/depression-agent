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

    response = requests.post(f"{API_BASE}/dm/step", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


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
    except Exception as exc:
        return f"❌ 报告生成失败：{exc}"


def user_step(
    message: str,
    history: List[Tuple[Optional[str], str]],
    session_id: str,
) -> Tuple[List[Tuple[Optional[str], str]], str, Dict[str, Any], str, Optional[str]]:
    message = message or ""
    text_payload = message.strip() or None
    risk_text = "无紧急风险提示。"
    progress: Dict[str, Any] = {}
    audio_value: Optional[str] = None

    try:
        result = _call_dm_step(session_id, text=text_payload, audio_ref=None)
    except Exception as exc:
        if text_payload:
            user_label = message
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
    history = history + [(user_label or None, assistant_reply)]

    progress = result.get("progress", {})
    risk_flag = result.get("risk_flag", False)
    risk_text = (
        "⚠️ 检测到高风险，请立即寻求紧急帮助。" if risk_flag else "无紧急风险提示。"
    )

    return history, risk_text, progress, session_id, audio_value


def initialize_conversation(
    session_id: str,
) -> Tuple[List[Tuple[Optional[str], str]], str, str, Dict[str, Any], Optional[str]]:
    """为新会话预拉取首个问题。"""

    sid = session_id or _init_session()
    history: List[Tuple[Optional[str], str]] = []
    risk_text = "无紧急风险提示。"
    progress: Dict[str, Any] = {}
    audio_value: Optional[str] = None

    try:
        result = _call_dm_step(sid)
    except Exception as exc:
        error_text = f"❌ 请求失败：{exc}"
        history.append((None, error_text))
        return history, sid, "⚠️ 初始化失败，请稍后重试。", {}, None

    assistant_reply = result.get("next_utterance", "")
    if assistant_reply:
        history.append((None, assistant_reply))

    progress = result.get("progress", {})
    risk_flag = result.get("risk_flag", False)
    risk_text = (
        "⚠️ 检测到高风险，请立即寻求紧急帮助。" if risk_flag else "无紧急风险提示。"
    )

    tts_url = result.get("tts_url")
    if tts_url:
        audio_value = tts_url

    audio_value = _ensure_audio_playable_url(sid, audio_value)
    return history, sid, risk_text, progress, audio_value


@dataclass(slots=True)
class TingwuStreamConfig:
    """听悟实时流配置"""
    appkey: str
    format: str = "pcm"
    language: str = "cn"
    sample_rate: int = 16000
    frame_ms: int = 20


class RealTimeTingwuClient:
    """优化的实时听悟客户端 - 专注于稳定性"""
    
    def __init__(self, complete_sentence_callback: Optional[Callable[[str], None]] = None):
        self.config = self._build_config()
        self.ws = None
        self.task_id = None
        self.is_connected = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.complete_sentence_queue = queue.Queue()
        self.running = False
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        self.current_sentence = ""
        self.last_audio_time = 0
        self.silence_packet = None
        self.keepalive_task = None
        self.complete_sentence_callback = complete_sentence_callback
        self.latest_result = "准备开始..."
        
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
        if audio_data.ndim == 1:
            return audio_data
        return np.mean(audio_data, axis=1)
    
    def _resample_audio(self, audio_data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        if original_sr == target_sr:
            return audio_data
        if audio_data.size == 0:
            return audio_data
            
        duration = len(audio_data) / original_sr
        target_length = int(duration * target_sr)
        if target_length == 0:
            return np.array([], dtype=np.float32)
            
        original_indices = np.arange(len(audio_data), dtype=np.float32)
        target_indices = np.linspace(0, len(audio_data)-1, target_length, dtype=np.float32)
        return np.interp(target_indices, original_indices, audio_data).astype(np.float32)
    
    def _audio_to_pcm(self, audio_data: np.ndarray) -> bytes:
        normalized = np.clip(audio_data, -1.0, 1.0)
        return (normalized * 32767).astype("<i2").tobytes()
    
    def _generate_silence_packet(self, duration_ms: int = 100) -> bytes:
        samples = int(self.config.sample_rate * duration_ms / 1000)
        silence_data = np.zeros(samples, dtype=np.float32)
        return self._audio_to_pcm(silence_data)
    
    async def connect(self):
        """连接到听悟WebSocket"""
        print("🎯 开始连接听悟服务")
        self.connection_attempts += 1
        
        try:
            print("🔄 正在创建听悟实时任务...")
            ws_url, self.task_id = await create_realtime_task()
            print(f"✅ 听悟任务创建成功: {self.task_id}")
            
            connect_kwargs = {
                "ping_interval": 10,
                "ping_timeout": 30,
                "max_size": 10 * 1024 * 1024,
            }
            
            try:
                print("🔄 正在建立WebSocket连接...")
                self.ws = await websockets.connect(ws_url, **connect_kwargs)
            except Exception as e:
                print(f"⚠️ 首次连接失败: {e}")
                if "SSL" in str(e) or "302" in str(e):
                    print("🔄 尝试使用非SSL连接...")
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
            
            print("📤 发送开始转录消息...")
            await self.ws.send(json.dumps(start_message, ensure_ascii=False))
            print("🚀 开始实时转录")
            return True
            
        except Exception as e:
            print(f"❌ 连接失败 (尝试 {self.connection_attempts}/{self.max_connection_attempts}): {e}")
            self.result_queue.put(f"❌ 连接失败: {e}")
            
            if self.connection_attempts >= self.max_connection_attempts:
                self.result_queue.put("❌ 连接失败次数过多，请检查网络和配置")
                return False
            return False
    
    async def send_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int):
        if not self.is_connected or not self.ws:
            return
        
        try:
            mono_audio = self._ensure_mono(audio_chunk.astype(np.float32))
            processed_audio = self._resample_audio(mono_audio, sample_rate, self.config.sample_rate)
            
            if processed_audio.size == 0:
                return
                
            pcm_data = self._audio_to_pcm(processed_audio)
            
            max_chunk_size = 16 * 1024
            if len(pcm_data) > max_chunk_size:
                for i in range(0, len(pcm_data), max_chunk_size):
                    chunk = pcm_data[i:i + max_chunk_size]
                    await self.ws.send(chunk)
                    self.last_audio_time = time.time()
                    await asyncio.sleep(0.001)
            else:
                await self.ws.send(pcm_data)
                self.last_audio_time = time.time()
                
        except Exception as e:
            print(f"⚠️ 发送音频失败: {e}")
    
    async def keepalive_loop(self):
        while self.running and self.is_connected:
            try:
                current_time = time.time()
                if current_time - self.last_audio_time > 3:
                    await self.send_keepalive_silence()
                    self.last_audio_time = current_time
                
                await asyncio.sleep(1)
            except Exception as e:
                print(f"⚠️ 保持连接循环错误: {e}")
                break
    
    async def send_keepalive_silence(self):
        if not self.is_connected or not self.ws or not self.silence_packet:
            return
        
        try:
            await self.ws.send(self.silence_packet)
        except Exception as e:
            print(f"⚠️ 发送静音包失败: {e}")
    
    async def receive_messages(self):
        if not self.is_connected or not self.ws:
            return
            
        try:
            async for message in self.ws:
                if not self.running:
                    break
                    
                if isinstance(message, bytes):
                    continue
                    
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON解析错误: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ 消息处理错误: {e}")
                    continue
                
                await self._handle_message(data)
                
        except websockets.exceptions.ConnectionClosed as e:
            if self.running:
                print(f"⚠️ WebSocket连接关闭: {e}")
                self.result_queue.put(f"⚠️ 连接中断: {e}")
        except Exception as e:
            if self.running:
                print(f"⚠️ 接收消息错误: {e}")
                self.result_queue.put(f"⚠️ 连接错误: {e}")
    
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
            word_texts = []
            for word in words:
                if isinstance(word, dict) and word.get("text"):
                    word_texts.append(word["text"])
            if word_texts:
                return " ".join(word_texts)
                
        sentences = payload.get("sentences")
        if isinstance(sentences, list) and sentences:
            sentence_texts = []
            for sentence in sentences:
                if isinstance(sentence, dict):
                    text = sentence.get("text")
                    if isinstance(text, str) and text.strip():
                        sentence_texts.append(text.strip())
            if sentence_texts:
                return " ".join(sentence_texts)
                
        return None
    
    async def _handle_message(self, data: dict):
        header = data.get("header", {})
        payload = data.get("payload", {})
        name = header.get("name")
        status = header.get("status")
        
        if isinstance(status, int) and status >= 40000000:
            error_msg = payload.get("message", payload.get("error_message", "未知错误"))
            detailed_error = f"❌ 听悟错误({status}): {error_msg}"
            print(detailed_error)
            self.latest_result = detailed_error
            self.result_queue.put(detailed_error)
            self.running = False
            return
        
        if name == "TranscriptionStarted":
            self.latest_result = "🎤 实时转录已开始，请说话..."
            self.result_queue.put(self.latest_result)
            
        elif name == "SentenceBegin":
            self.current_sentence = ""
            self.latest_result = "🔊 检测到语音开始"
            self.result_queue.put(self.latest_result)
            
        elif name in ["TranscriptionResult", "TranscriptionResultChanged"]:
            result_text = self._extract_text_from_payload(payload)
            if result_text:
                self.current_sentence = result_text
                display_result = result_text
                if len(display_result) > 200:
                    display_result = display_result[:200] + "..."
                prefix = "📝" if name == "TranscriptionResult" else "🔄"
                self.latest_result = f"{prefix} {display_result}"
                self.result_queue.put(self.latest_result)
                
        elif name == "SentenceEnd":
            result_text = self._extract_text_from_payload(payload)
            if not result_text and self.current_sentence:
                result_text = self.current_sentence
                
            if result_text:
                self.complete_sentence_queue.put(result_text)
                if self.complete_sentence_callback:
                    self.complete_sentence_callback(result_text)
                
                print(f"🎯 [完整句子] {result_text}")
                
                self.latest_result = f"✅ 完整句子: {result_text}"
                self.result_queue.put(self.latest_result)
                self.current_sentence = ""
            
        elif name == "TranscriptionCompleted":
            self.latest_result = "🏁 转录完成"
            self.result_queue.put(self.latest_result)
            
        elif name == "TaskFailed":
            error_msg = payload.get("message", payload.get("error_message", "任务执行失败"))
            detailed_error = f"❌ 任务失败: {error_msg}"
            print(detailed_error)
            self.latest_result = detailed_error
            self.result_queue.put(detailed_error)
            self.running = False
            
        elif name == "SpeechStartDetected":
            self.latest_result = "🔊 检测到语音开始"
            self.result_queue.put(self.latest_result)
            
        elif name == "SpeechEndDetected":
            self.latest_result = "🔇 检测到语音结束"
            self.result_queue.put(self.latest_result)
    
    async def start_realtime_stream(self):
        """开始实时流式处理"""
        print("🎬 开始实时流处理")
        self.running = True
        
        if not await self.connect():
            self.running = False
            print("❌ 连接失败，停止实时流处理")
            return
        
        print("✅ 连接成功，启动消息接收和保活任务")
        
        receive_task = asyncio.create_task(self.receive_messages())
        self.keepalive_task = asyncio.create_task(self.keepalive_loop())
        
        try:
            print("🔄 进入音频处理循环")
            while self.running:
                try:
                    audio_data = self.audio_queue.get_nowait()
                    sample_rate, audio_chunk = audio_data
                    
                    if audio_chunk.size > 0:
                        await self.send_audio_chunk(audio_chunk, sample_rate)
                    
                except queue.Empty:
                    await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"⚠️ 处理音频数据错误: {e}")
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            print(f"❌ 实时流错误: {e}")
            self.result_queue.put(f"❌ 实时流错误: {e}")
        finally:
            print("🛑 结束实时流处理")
            self.running = False
            if not receive_task.done():
                receive_task.cancel()
            if self.keepalive_task and not self.keepalive_task.done():
                self.keepalive_task.cancel()
            
            with contextlib.suppress(asyncio.CancelledError):
                if not receive_task.done():
                    await receive_task
                if self.keepalive_task and not self.keepalive_task.done():
                    await self.keepalive_task
            
            await self.close()
    
    def add_audio_data(self, sample_rate: int, audio_data: np.ndarray):
        if self.running and audio_data is not None and audio_data.size > 0:
            self.audio_queue.put((sample_rate, audio_data))
    
    def get_latest_result(self) -> str:
        """获取最新结果，避免频繁队列操作"""
        return self.latest_result
    
    def get_results(self):
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results
    
    def get_complete_sentences(self) -> List[str]:
        sentences = []
        while not self.complete_sentence_queue.empty():
            try:
                sentences.append(self.complete_sentence_queue.get_nowait())
            except queue.Empty:
                break
        return sentences
    
    async def close(self):
        self.running = False
        
        if self.is_connected and self.ws:
            try:
                stop_message = {
                    "header": {
                        "namespace": "SpeechTranscriber", 
                        "name": "StopTranscription"
                    },
                    "payload": {}
                }
                await self.ws.send(json.dumps(stop_message, ensure_ascii=False))
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"⚠️ 发送停止消息时出错: {e}")
            finally:
                try:
                    await self.ws.close()
                except Exception as e:
                    print(f"⚠️ 关闭WebSocket时出错: {e}")
                self.is_connected = False
                print("🔚 WebSocket连接已关闭")
        
        if self.task_id:
            try:
                await stop_realtime_task(self.task_id)
                print("🔚 听悟任务已停止")
            except Exception as e:
                print(f"⚠️ 停止任务失败: {e}")


# 全局客户端实例和状态管理
_client: Optional[RealTimeTingwuClient] = None
_client_lock = threading.Lock()
_complete_sentences: List[str] = []
_oss_client = OSSClient()

# 状态标志
_transcription_active = False


def handle_complete_sentence(sentence: str) -> None:
    """处理完整句子的回调函数"""
    global _complete_sentences
    _complete_sentences.append(sentence)
    print(f"🎯 [完整句子] {sentence}")


def get_latest_complete_sentence() -> Optional[str]:
    """获取最新的完整句子"""
    global _complete_sentences
    if _complete_sentences:
        return _complete_sentences[-1]
    return None


def get_all_complete_sentences() -> List[str]:
    """获取所有完整句子"""
    global _complete_sentences
    return _complete_sentences.copy()


def clear_complete_sentences() -> None:
    """清空完整句子列表"""
    global _complete_sentences
    _complete_sentences.clear()


def _format_sentences_display(sentences: List[str]) -> str:
    """格式化句子显示"""
    if not sentences:
        return "暂无完整句子"
    return "\n\n".join(f"{index + 1}. {value}" for index, value in enumerate(sentences))


def _ensure_audio_playable_url(session_id: str, audio_value: Optional[str]) -> Optional[str]:
    """确保音频URL可播放"""
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


# 创建全局事件循环和任务管理
_event_loop = None
_background_tasks = set()

def get_or_create_event_loop():
    """获取或创建事件循环"""
    global _event_loop
    if _event_loop is None:
        _event_loop = asyncio.new_event_loop()
        # 在新线程中运行事件循环
        def run_loop():
            asyncio.set_event_loop(_event_loop)
            _event_loop.run_forever()
        
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
    return _event_loop

def run_async_in_background(coro):
    """在后台运行异步协程"""
    loop = get_or_create_event_loop()
    task = asyncio.run_coroutine_threadsafe(coro, loop)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

async def start_realtime_transcription():
    """开始实时转录 - 异步版本"""
    global _client, _transcription_active
    
    print("🚀 启动实时转录服务")
    
    with _client_lock:
        if _client is None or not _client.running:
            _client = RealTimeTingwuClient(complete_sentence_callback=handle_complete_sentence)
    
    if _client:
        _transcription_active = True
        # 在后台启动转录流
        run_async_in_background(_client.start_realtime_stream())
        return "🔄 正在启动实时转录..."
    
    return "❌ 无法启动客户端"


async def stop_transcription() -> str:
    """停止转录"""
    global _client, _transcription_active
    
    print("🛑 停止转录服务")
    
    _transcription_active = False
    
    with _client_lock:
        client = _client
        _client = None
    
    if client is not None:
        await client.close()
    
    clear_complete_sentences()
    return "🛑 转录已停止"


def get_realtime_status() -> str:
    """获取实时转录状态"""
    global _client
    with _client_lock:
        if _client is not None and _client.running:
            return _client.get_latest_result()
    return "实时转录未启动"


def get_current_sentences() -> Tuple[str, str]:
    """获取当前句子状态"""
    latest = get_latest_complete_sentence() or "暂无完整句子"
    all_sentences = _format_sentences_display(get_all_complete_sentences())
    return latest, all_sentences


def start_transcription_sync():
    """同步版本的开始转录函数"""
    try:
        # 在后台线程中运行异步函数
        run_async_in_background(start_realtime_transcription())
        return "🔄 正在启动实时转录..."
    except Exception as e:
        print(f"❌ 启动转录失败: {e}")
        return f"❌ 启动失败: {e}"


def stop_transcription_sync():
    """同步版本的停止转录函数"""
    try:
        # 在后台线程中运行异步函数
        run_async_in_background(stop_transcription())
        return "🛑 正在停止转录..."
    except Exception as e:
        print(f"❌ 停止转录失败: {e}")
        return f"❌ 停止失败: {e}"


def handle_audio_input(audio: Optional[Tuple[int, np.ndarray]]) -> str:
    """处理音频输入 - 完全同步版本"""
    global _client
    
    if audio is None:
        return get_realtime_status()
    
    try:
        sample_rate, audio_data = audio
        
        if audio_data is None or audio_data.size == 0:
            return "未检测到有效音频数据"
        
        # 添加音频数据到客户端
        with _client_lock:
            if _client is not None and _client.running:
                _client.add_audio_data(sample_rate, audio_data)
                return _client.get_latest_result()
            else:
                return "请先点击'开始录音'按钮启动语音识别"
                
    except Exception as e:
        print(f"❌ 处理音频输入错误: {e}")
        return f"❌ 处理错误: {e}"


# 轮询状态函数
def poll_realtime_status():
    """轮询实时状态"""
    return get_realtime_status()

def poll_sentences():
    """轮询句子状态"""
    return get_current_sentences()


def build_ui() -> gr.Blocks:
    """优化的 Gradio 界面 - 专注于稳定性"""
    with gr.Blocks(theme=gr.themes.Soft(), title="抑郁随访助手") as demo:
        session_state = gr.State(_init_session())

        gr.Markdown(
            """
            # 抑郁随访助手
            集成文本/音频问答与实时语音识别，完成自动随访与报告。
            """
        )

        with gr.Tabs():
            with gr.Tab("评估"):
                chatbot = gr.Chatbot(height=420, label="对话记录", show_copy_button=True)

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
                        
                        with gr.Row():
                            send_button = gr.Button("发送文本", variant="primary")
                            clear_chat_btn = gr.Button("清空对话", variant="secondary")
                        
                        gr.Markdown(
                            "提示：可手动输入文本，系统会同步更新问答。"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 🎧 实时语音识别")
                        
                        with gr.Row():
                            start_mic_btn = gr.Button("🎤 开始录音", variant="primary")
                            stop_mic_btn = gr.Button("🛑 停止录音", variant="stop")
                        
                        realtime_output = gr.Textbox(
                            label="实时识别状态",
                            lines=4,
                            show_copy_button=True,
                            interactive=False,
                            value="点击'开始录音'启动语音识别"
                        )
                        
                        # 添加音频输入组件
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            streaming=True,
                            label="实时音频输入",
                            show_download_button=False,
                            show_share_button=False,
                        )
                        
                        with gr.Accordion("📝 完整句子", open=True):
                            latest_sentence = gr.Textbox(
                                label="最新完整句子",
                                lines=2,
                                interactive=False,
                                placeholder="暂无完整句子",
                            )
                            all_sentences = gr.Textbox(
                                label="全部完整句子",
                                lines=4,
                                interactive=False,
                                placeholder="暂无完整句子",
                            )
                            with gr.Row():
                                submit_sentence_btn = gr.Button("🚀 提交此句子", variant="primary")
                                refresh_btn = gr.Button("🔄 刷新", size="sm")
                                clear_btn = gr.Button("🗑️ 清空", size="sm")
                        
                        gr.Markdown(
                            """
                            **使用说明:**
                            1. 点击 **开始录音** 启动语音识别
                            2. 说话时可以看到实时转录结果
                            3. 完整句子会自动显示在上方区域
                            4. 点击 **提交此句子** 将句子发送给AI助手
                            5. 点击 **停止录音** 结束识别
                            """
                        )

            with gr.Tab("报告"):
                gr.Markdown("## 生成评估报告")
                gr.Markdown("点击按钮后将在 /tmp/depression_agent_reports/ 下生成 PDF。")
                report_button = gr.Button("生成报告")
                report_status = gr.Markdown("等待生成指令…")

        # 文本输入处理
        def _on_submit(
            message: str,
            history: List[Tuple[Optional[str], str]],
            session_id: str,
        ) -> Tuple[
            List[Tuple[Optional[str], str]],
            str,
            str,
            str,
            Dict[str, Any],
            Optional[str],
        ]:
            chat, risk_text, progress, sid, audio_value = user_step(
                message, history, session_id
            )
            playable_audio = _ensure_audio_playable_url(sid, audio_value)
            return chat, "", sid, risk_text, progress, playable_audio

        text_input.submit(
            _on_submit,
            inputs=[text_input, chatbot, session_state],
            outputs=[
                chatbot,
                text_input,
                session_state,
                risk_alert,
                progress_display,
                audio_sys,
            ],
        )

        send_button.click(
            _on_submit,
            inputs=[text_input, chatbot, session_state],
            outputs=[
                chatbot,
                text_input,
                session_state,
                risk_alert,
                progress_display,
                audio_sys,
            ],
        )

        # 清空对话
        def clear_chat():
            return [], _init_session(), "无紧急风险提示。", {}, None

        clear_chat_btn.click(
            fn=clear_chat,
            outputs=[chatbot, session_state, risk_alert, progress_display, audio_sys]
        )

        # 实时语音识别控制
        start_mic_btn.click(
            fn=start_transcription_sync,
            outputs=[realtime_output]
        )

        stop_mic_btn.click(
            fn=stop_transcription_sync,
            outputs=[realtime_output]
        )

        # 音频输入处理
        audio_input.stream(
            fn=handle_audio_input,
            inputs=[audio_input],
            outputs=[realtime_output],
            show_progress="hidden"
        )

        # 刷新句子
        refresh_btn.click(
            fn=poll_sentences,
            outputs=[latest_sentence, all_sentences]
        )

        # 清空句子
        def clear_sentences_action() -> Tuple[str, str]:
            clear_complete_sentences()
            return "已清空", "已清空"

        clear_btn.click(
            fn=clear_sentences_action,
            outputs=[latest_sentence, all_sentences],
        )

        # 提交句子进行问答
        def submit_current_sentence_sync(
            history: List[Tuple[Optional[str], str]],
            session_id: str,
        ) -> Tuple[
            List[Tuple[Optional[str], str]],
            str,
            Dict[str, Any],
            str,
            Optional[str],
            str,  # latest_sentence
            str,  # all_sentences
        ]:
            current_sentence = get_latest_complete_sentence()
            if not current_sentence or current_sentence == "暂无完整句子":
                # 如果没有句子，返回当前状态
                latest, all_sents = get_current_sentences()
                return history, "无紧急风险提示。", {}, session_id, None, latest, all_sents
            
            # 提交句子进行问答
            updated_history, risk_text, progress, updated_session, audio_value = user_step(
                current_sentence, history, session_id
            )
            
            # 获取更新后的句子状态
            latest, all_sents = get_current_sentences()
            
            return updated_history, risk_text, progress, updated_session, audio_value, latest, all_sents

        submit_sentence_btn.click(
            fn=submit_current_sentence_sync,
            inputs=[chatbot, session_state],
            outputs=[
                chatbot,
                risk_alert,
                progress_display,
                session_state,
                audio_sys,
                latest_sentence,
                all_sentences,
            ]
        )

        report_button.click(
            lambda sid: _generate_report(sid),
            inputs=[session_state],
            outputs=[report_status],
        )

        # 初始化对话
        demo.load(
            fn=initialize_conversation,
            inputs=[session_state],
            outputs=[chatbot, session_state, risk_alert, progress_display, audio_sys],
        )

        # 添加轮询组件
        with gr.Row(visible=False) as poll_row:
            status_poll_trigger = gr.Button("更新状态", elem_id="status_poll")
            sentences_poll_trigger = gr.Button("更新句子", elem_id="sentences_poll")

        # 状态轮询
        status_poll_trigger.click(
            fn=poll_realtime_status,
            outputs=[realtime_output]
        )

        # 句子轮询
        sentences_poll_trigger.click(
            fn=poll_sentences,
            outputs=[latest_sentence, all_sentences]
        )

        # 添加JavaScript代码来实现定时轮询
        demo.load(
            None,
            None,
            None,
            js="""
            () => {
                // 状态轮询 - 每秒更新一次
                setInterval(() => {
                    const statusBtn = document.getElementById('status_poll');
                    if (statusBtn) statusBtn.click();
                }, 1000);
                
                // 句子轮询 - 每2秒更新一次  
                setInterval(() => {
                    const sentencesBtn = document.getElementById('sentences_poll');
                    if (sentencesBtn) sentencesBtn.click();
                }, 2000);
            }
            """
        )

    return demo


if __name__ == "__main__":
    print("🚀 启动抑郁随访助手应用...")
    print(f"📁 项目根目录: {PROJECT_ROOT}")
    print(f"🔑 听悟 AppKey: {settings.TINGWU_APPKEY or settings.ALIBABA_TINGWU_APPKEY or '未配置'}")
    
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )

