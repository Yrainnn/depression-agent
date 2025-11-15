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

DEFAULT_PROGRESS_TOTAL = 17

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
    if not isinstance(history, list):
        history = []
    risk_text = "无紧急风险提示。"
    progress: Dict[str, Any] = {}
    media_value: Optional[str] = None

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
    media_value = _extract_media_value(session_id, result)

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

    return history, risk_text, progress, session_id, media_value

def initialize_conversation(
    session_id: str,
) -> Tuple[List[Tuple[Optional[str], str]], str, str, Dict[str, Any], Optional[str]]:
    """为新会话预拉取首个问题。"""

    sid = session_id or _init_session()
    history: List[Tuple[Optional[str], str]] = []
    risk_text = "无紧急风险提示。"
    progress: Dict[str, Any] = {}

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

    media_value = _extract_media_value(sid, result)
    return history, sid, risk_text, progress, media_value

@dataclass(slots=True)
class TingwuStreamConfig:
    """听悟实时流配置"""
    appkey: str
    format: str = "pcm"
    language: str = "cn"
    sample_rate: int = 16000
    frame_ms: int = 20

class RealTimeTingwuClient:
    """简化的实时听悟客户端"""
    
    def __init__(self, complete_sentence_callback: Optional[Callable[[str], None]] = None):
        self.config = self._build_config()
        self.ws = None
        self.task_id = None
        self.is_connected = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.complete_sentence_queue = queue.Queue()
        self.running = False
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
    
    async def connect(self):
        """连接到听悟WebSocket"""
        try:
            ws_url, self.task_id = await create_realtime_task()
            
            connect_kwargs = {
                "ping_interval": 10,
                "ping_timeout": 30,
                "max_size": 10 * 1024 * 1024,
            }
            
            try:
                self.ws = await websockets.connect(ws_url, **connect_kwargs)
            except Exception as e:
                if "SSL" in str(e) or "302" in str(e):
                    connect_kwargs["ssl"] = ssl._create_unverified_context()
                    self.ws = await websockets.connect(ws_url, **connect_kwargs)
                else:
                    raise
            
            self.is_connected = True
            
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
            return True
            
        except Exception as e:
            self.result_queue.put(f"❌ 连接失败: {e}")
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
            await self.ws.send(pcm_data)
                
        except Exception as e:
            print(f"⚠️ 发送音频失败: {e}")
    
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
                except json.JSONDecodeError:
                    continue
                
                await self._handle_message(data)
                
        except websockets.exceptions.ConnectionClosed:
            if self.running:
                self.result_queue.put("⚠️ 连接中断")
        except Exception as e:
            if self.running:
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
                
        return None
    
    async def _handle_message(self, data: dict):
        header = data.get("header", {})
        payload = data.get("payload", {})
        name = header.get("name")
        
        if name == "TranscriptionStarted":
            self.latest_result = "🎤 实时转录已开始，请说话..."
            self.result_queue.put(self.latest_result)
            
        elif name in ["TranscriptionResult", "TranscriptionResultChanged"]:
            result_text = self._extract_text_from_payload(payload)
            if result_text:
                self.latest_result = f"📝 {result_text}"
                self.result_queue.put(self.latest_result)
                
        elif name == "SentenceEnd":
            result_text = self._extract_text_from_payload(payload)
            if result_text:
                self.complete_sentence_queue.put(result_text)
                if self.complete_sentence_callback:
                    self.complete_sentence_callback(result_text)
                
                self.latest_result = f"✅ 完整句子: {result_text}"
                self.result_queue.put(self.latest_result)
            
        elif name == "TaskFailed":
            error_msg = payload.get("message", payload.get("error_message", "任务执行失败"))
            detailed_error = f"❌ 任务失败: {error_msg}"
            self.latest_result = detailed_error
            self.result_queue.put(detailed_error)
            self.running = False
    
    async def start_realtime_stream(self):
        """开始实时流式处理"""
        self.running = True
        
        if not await self.connect():
            self.running = False
            return
        
        receive_task = asyncio.create_task(self.receive_messages())
        
        try:
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
            self.result_queue.put(f"❌ 实时流错误: {e}")
        finally:
            self.running = False
            if not receive_task.done():
                receive_task.cancel()
            
            with contextlib.suppress(asyncio.CancelledError):
                if not receive_task.done():
                    await receive_task
            
            await self.close()
    
    def add_audio_data(self, sample_rate: int, audio_data: np.ndarray):
        if self.running and audio_data is not None and audio_data.size > 0:
            self.audio_queue.put((sample_rate, audio_data))
    
    def get_latest_result(self) -> str:
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
            except Exception:
                pass
            finally:
                try:
                    await self.ws.close()
                except Exception:
                    pass
                self.is_connected = False

# 全局客户端实例和状态管理
_client: Optional[RealTimeTingwuClient] = None
_client_lock = threading.Lock()
_complete_sentences: List[str] = []
_transcription_active = False

def handle_complete_sentence(sentence: str) -> None:
    global _complete_sentences
    _complete_sentences.append(sentence)

def get_latest_complete_sentence() -> Optional[str]:
    global _complete_sentences
    if _complete_sentences:
        return _complete_sentences[-1]
    return None

def get_all_complete_sentences() -> List[str]:
    global _complete_sentences
    return _complete_sentences.copy()

def clear_complete_sentences() -> None:
    global _complete_sentences
    _complete_sentences.clear()

def _format_sentences_display(sentences: List[str]) -> str:
    if not sentences:
        return "暂无完整句子"
    return "\n\n".join(f"{index + 1}. {value}" for index, value in enumerate(sentences))

def _ensure_audio_playable_url(session_id: str, audio_value: Optional[str]) -> Optional[str]:
    if not audio_value:
        return None
    if isinstance(audio_value, str) and audio_value.startswith(("http://", "https://")):
        return audio_value
    local_path = audio_value
    if isinstance(local_path, str) and local_path.startswith("file://"):
        local_path = local_path[7:]
    if not local_path or not Path(local_path).exists():
        return audio_value
    return str(Path(local_path).resolve())

def _extract_media_value(session_id: str, result: Dict[str, Any]) -> Optional[str]:
    if not result:
        return None

    media_type = (result.get("media_type") or "").lower()
    if media_type == "video":
        return result.get("video_url")

    if media_type == "audio":
        return _ensure_audio_playable_url(session_id, result.get("tts_url"))

    return None

# 创建全局事件循环和任务管理
_event_loop = None
_background_tasks = set()

def get_or_create_event_loop():
    global _event_loop
    if _event_loop is None:
        _event_loop = asyncio.new_event_loop()
        def run_loop():
            asyncio.set_event_loop(_event_loop)
            _event_loop.run_forever()
        
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
    return _event_loop

def run_async_in_background(coro):
    loop = get_or_create_event_loop()
    task = asyncio.run_coroutine_threadsafe(coro, loop)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

async def start_realtime_transcription():
    global _client, _transcription_active
    
    with _client_lock:
        if _client is None or not _client.running:
            _client = RealTimeTingwuClient(complete_sentence_callback=handle_complete_sentence)
    
    if _client:
        _transcription_active = True
        run_async_in_background(_client.start_realtime_stream())
        return "🔄 正在启动实时转录..."
    
    return "❌ 无法启动客户端"

async def stop_transcription() -> str:
    global _client, _transcription_active
    
    _transcription_active = False
    
    with _client_lock:
        client = _client
        _client = None
    
    if client is not None:
        await client.close()
    
    clear_complete_sentences()
    return "🛑 转录已停止"

def get_realtime_status() -> str:
    global _client
    with _client_lock:
        if _client is not None and _client.running:
            return _client.get_latest_result()
    return "实时转录未启动"

def get_current_sentences() -> Tuple[str, str]:
    latest = get_latest_complete_sentence() or "暂无完整句子"
    all_sentences = _format_sentences_display(get_all_complete_sentences())
    return latest, all_sentences

def start_transcription_sync():
    try:
        run_async_in_background(start_realtime_transcription())
        return "🔄 正在启动实时转录..."
    except Exception as e:
        return f"❌ 启动失败: {e}"

def stop_transcription_sync():
    try:
        run_async_in_background(stop_transcription())
        return "🛑 正在停止转录..."
    except Exception as e:
        return f"❌ 停止失败: {e}"

def handle_audio_input(audio: Optional[Tuple[int, np.ndarray]]) -> str:
    global _client
    
    if audio is None:
        return get_realtime_status()
    
    try:
        sample_rate, audio_data = audio
        
        if audio_data is None or audio_data.size == 0:
            return "未检测到有效音频数据"
        
        with _client_lock:
            if _client is not None and _client.running:
                _client.add_audio_data(sample_rate, audio_data)
                return _client.get_latest_result()
            else:
                return "请先点击'开始录音'按钮启动语音识别"
                
    except Exception as e:
        return f"❌ 处理错误: {e}"

def poll_realtime_status():
    return get_realtime_status()

def poll_sentences():
    return get_current_sentences()

def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0

def _format_progress_heading(progress: Optional[Dict[str, Any]]) -> str:
    data = progress or {}

    index = 0
    total = 0

    if "index" in data or "total" in data:
        index = _coerce_int(data.get("index"))
        total = _coerce_int(data.get("total"))

    if total <= 0 and ("current_step" in data or "total_steps" in data):
        index = _coerce_int(data.get("current_step"))
        total = _coerce_int(data.get("total_steps"))

    if total <= 0:
        total = DEFAULT_PROGRESS_TOTAL

    if index < 0:
        index = 0
    if total > 0 and index > total:
        index = total

    pct = _get_progress_value(data)
    if pct <= 0 and total > 0 and index > 0:
        pct = (index / total) * 100.0

    pct = max(0.0, min(100.0, pct))

    return f"### 🧠 专业评估进度（{index}/{total} · {pct:.0f}%）"

def _get_progress_value(progress: Dict[str, Any]) -> float:
    if not progress:
        return 0.0
    
    if "index" in progress and "total" in progress:
        index = progress["index"]
        total = progress["total"]
        if total > 0:
            return (index / total) * 100.0
    
    if "overall" in progress:
        value = progress["overall"]
        if isinstance(value, (int, float)):
            return float(value)
    
    if "progress" in progress:
        value = progress["progress"]
        if isinstance(value, (int, float)):
            return float(value)
    
    if "percentage" in progress:
        value = progress["percentage"]
        if isinstance(value, (int, float)):
            return float(value)
    
    if "current_step" in progress and "total_steps" in progress:
        current = progress["current_step"]
        total = progress["total_steps"]
        if total > 0:
            return (current / total) * 100.0
    
    return 0.0

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        title="智能心境健康评估系统",
        css="""
        .gradio-container {
            max-width: 100% !important;
            width: 100% !important;
            background-image: url("https://bpic.588ku.com/back_pic/06/42/63/94647d8c29b3127.jpg") !important;
            background-color: #e8f4f8 !important;
            background-repeat: no-repeat !important;
            background-position: center center !important;
            background-size: cover !important;
            background-attachment: fixed !important;
            min-height: 100vh !important;
        }
        .risk-alert {
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            font-weight: bold;
        }
        .risk-high {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            border: 2px solid #ff3838;
        }
        .risk-low {
            background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
            color: white;
            border: 2px solid #00b894;
        }
        .section-title {
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 1.3em;
            margin-bottom: 12px;
        }
        .progress-title {
            text-align: center;
            margin: 8px 0 16px;
            font-weight: 700;
            color: #374151;
        }
        .gr-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
        }
        .gr-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .transparent-markdown {
            background: transparent !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            border: none !important;
            box-shadow: none !important;
        }
        .title-container {
            background: transparent !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            border: none !important;
            box-shadow: none !important;
            padding: 20px !important;
            margin-bottom: 20px !important;
        }

        #video_sys,
        #chatbot,
        #patient_input,
        #realtime_status,
        #audio_capture,
        #latest_sentence,
        #all_sentences{
            border: 1.5px solid #cbd5e1 !important;
            border-radius: 10px !important;
            background: rgba(255, 255, 255, 0.96) !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.12) !important;
        }
        
        #video_sys {
            width: 238px !important;
            height: 420px !important;
            overflow: hidden !important;
        }

        #video_sys video {
            width: 100% !important;
            height: 100% !important;
            object-fit: cover !important;
            border-radius: 10px;
        }

        #chatbot {
            height: 420px !important;
        }

        .gradio-container .gr-row {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        """
    ) as demo:
        session_state = gr.State(_init_session())
        last_video_state = gr.State(None)
        
        gr.Markdown(
            """
            <div class="title-container">
                <div style="text-align: center; padding: 10px 0;">
                    <h1 style="
                        background: linear-gradient(135deg, #20e3b2 0%, #0f3d5e 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-weight: 800;
                        font-size: 3em;
                        margin-bottom: 8px;
                    ">
                        🧠 智能心境健康评估系统
                    </h1>
                    <p style="
                        color: #666;
                        font-size: 1.1em;
                        margin-bottom: 15px;
                        font-weight: 500;
                    ">
                        融合多模态交互的精准心理健康筛查与智能随访平台
                    </p>
                </div>
            </div>
            """,
            elem_classes=["transparent-markdown"],
        )

        with gr.Tabs():
            with gr.Tab("🏥 专业评估"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=7, min_width=600):
                        gr.Markdown(
                            """
                            <div class="section-title">💡 专业提示</div>
                            <ul style="margin:8px 0 0 18px; font-weight:700;line-height:1.6;">
                              <li>请尽可能详细描述您的真实感受</li>
                              <li>系统会严格保密您的所有信息</li>
                              <li>如需紧急帮助，请立即联系专业医生</li>
                            </ul>
                            """,
                            elem_classes=["transparent-markdown"],
                        )

                        progress_title = gr.Markdown(
                            _format_progress_heading(None),
                            elem_classes=["progress-title", "transparent-markdown"],
                        )

                        with gr.Row(equal_height=True):
                            with gr.Column(scale=3, min_width=240):
                                video_sys = gr.Video(
                                    label="数字人交互",
                                    interactive=False,
                                    autoplay=True,
                                    elem_id="video_sys",
                                    height=420,
                                )

                            with gr.Column(scale=7, min_width=600):
                                chatbot = gr.Chatbot(
                                    height=420,
                                    label="智能对话记录",
                                    show_copy_button=True,
                                    elem_id="chatbot",
                                )

                        risk_alert = gr.Markdown(
                            """
                            <div class="risk-alert risk-low">
                                ✅ 当前状态：无紧急风险提示
                            </div>
                            """,
                            elem_classes=["transparent-markdown"],
                        )

                        text_input = gr.Textbox(
                            label="患者自述输入",
                            placeholder="请详细描述您近期的情绪状态、睡眠质量、生活压力等情况...",
                            lines=4,
                            max_lines=6,
                            show_copy_button=True,
                            elem_id="patient_input",
                        )

                        with gr.Row():
                            send_button = gr.Button("📤 提交评估", variant="primary", size="lg")
                            clear_chat_btn = gr.Button("🔄 重新开始", variant="secondary")

                    with gr.Column(scale=4, min_width=400):
                        gr.Markdown(
                            """
                            <div class="section-title">🎯 语音使用指南</div>
                            <ol style="margin:8px 0 0 18px; font-weight:700;line-height:1.6;">
                              <li>点击 <b>开始语音输入</b> 启动语音识别</li>
                              <li>用自然语言描述您的情况</li>
                              <li>识别结果会实时显示，完整句子会自动汇总</li>
                              <li>需要时点击 <b>停止录音</b> 结束语音输入</li>
                            </ol>
                            """,
                            elem_classes=["transparent-markdown"],
                        )

                        gr.Markdown(
                            """
                            <div class="section-title">
                                🎤 智能语音识别
                            </div>
                            """,
                            elem_classes=["transparent-markdown"],
                        )
                        
                        with gr.Row():
                            start_mic_btn = gr.Button(
                                "🎙️ 开始语音输入", 
                                variant="primary",
                                size="lg"
                            )
                            stop_mic_btn = gr.Button(
                                "⏹️ 停止录音", 
                                variant="stop"
                            )
                        
                        realtime_output = gr.Textbox(
                            label="实时转录状态",
                            lines=3,
                            show_copy_button=True,
                            interactive=False,
                            value="点击上方按钮开始语音输入",
                            placeholder="语音识别结果将实时显示在这里...",
                            elem_id="realtime_status",
                        )

                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            streaming=True,
                            label="实时音频采集",
                            elem_id="audio_capture",
                        )

                        with gr.Accordion("📝 语音识别结果", open=True):
                            latest_sentence = gr.Textbox(
                                label="最新识别内容",
                                lines=2,
                                interactive=False,
                                placeholder="完整句子将自动显示在这里...",
                                show_copy_button=True,
                                elem_id="latest_sentence",
                            )
                            all_sentences = gr.Textbox(
                                label="历史识别记录",
                                lines=3,
                                interactive=False,
                                placeholder="所有识别结果将汇总在这里...",
                                show_copy_button=True,
                                elem_id="all_sentences",
                            )
                            with gr.Row():
                                submit_sentence_btn = gr.Button(
                                    "🚀 提交此内容", 
                                    variant="primary",
                                    size="sm"
                                )
                                refresh_btn = gr.Button("🔄 刷新", size="sm")
                                clear_btn = gr.Button("🗑️ 清空记录", size="sm")
                        
            with gr.Tab("📊 评估报告"):
                gr.Markdown(
                    """
                    <div class="section-title">
                        📈 专业评估报告
                    </div>
                    """,
                    elem_classes=["transparent-markdown"],
                )
                gr.Markdown(
                    """
                    **系统将基于对话内容生成专业评估报告，包括：**
                    - 📋 综合心理状态分析
                    - 📊 风险评估等级
                    - 💡 个性化建议方案
                    - 🏥 专业转诊指引
                    """,
                    elem_classes=["transparent-markdown"],
                )
                with gr.Row():
                    report_button = gr.Button("生成专业报告", variant="primary", size="lg")
                report_status = gr.Markdown("等待生成指令…")

        # 修改文本输入处理函数 - 修复视频闪屏问题
        def _on_submit(
            message: str,
            last_video: Optional[str],
            history: List[Tuple[Optional[str], str]],
            session_id: str,
        ) -> Tuple[
            List[Tuple[Optional[str], str]],
            str,
            str,
            str,
            Optional[str],
            Optional[str],
            gr.update,
        ]:
            chat, risk_text, progress, sid, media_value = user_step(
                message, history, session_id
            )

            progress_text = _format_progress_heading(progress)

            if "高风险" in risk_text:
                risk_display = f"""
                <div class="risk-alert risk-high">
                    ⚠️ {risk_text}
                </div>
                """
            else:
                risk_display = f"""
                <div class="risk-alert risk-low">
                    ✅ {risk_text}
                </div>
                """
            
            # 🎯 关键修改：始终保留上一个视频，除非有明确的新视频
            if media_value is None:
                # 没有新视频时，保持显示上一个视频
                media_value = last_video
            else:
                # 有新视频时，更新状态
                last_video = media_value
            
            # 确保 media_value 是字符串类型
            if not isinstance(media_value, (str, type(None))):
                media_value = None
                
            return (
                chat,
                "",
                sid,
                risk_display,
                media_value,
                last_video,
                gr.update(value=progress_text),
            )

        text_input.submit(
            _on_submit,
            inputs=[text_input, last_video_state, chatbot, session_state],
            outputs=[
                chatbot,
                text_input,
                session_state,
                risk_alert,
                video_sys,
                last_video_state,
                progress_title,
            ],
        )

        send_button.click(
            _on_submit,
            inputs=[text_input, last_video_state, chatbot, session_state],
            outputs=[
                chatbot,
                text_input,
                session_state,
                risk_alert,
                video_sys,
                last_video_state,
                progress_title,
            ],
        )

        # 修改清空对话函数
        def clear_chat():
            new_session = _init_session()
            return (
                [],
                new_session,
                """
            <div class="risk-alert risk-low">
                ✅ 当前状态：无紧急风险提示
            </div>
            """,
                None,
                None,
                gr.update(value=_format_progress_heading(None)),
            )

        clear_chat_btn.click(
            fn=clear_chat,
            outputs=[chatbot, session_state, risk_alert, video_sys, last_video_state, progress_title]
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
            return "已清空记录", "已清空记录"

        clear_btn.click(
            fn=clear_sentences_action,
            outputs=[latest_sentence, all_sentences],
        )

        # 修改提交句子函数 - 修复视频闪屏问题
        def submit_current_sentence_sync(
            history: List[Tuple[Optional[str], str]],
            session_id: str,
            last_video: Optional[str],
        ) -> Tuple[
            List[Tuple[Optional[str], str]],
            str,
            str,
            Optional[str],
            Optional[str],
            str,
            str,
            gr.update,
        ]:
            current_sentence = get_latest_complete_sentence()
            if not current_sentence or current_sentence == "暂无完整句子":
                latest, all_sents = get_current_sentences()
                return (
                    history,
                    """
                <div class="risk-alert risk-low">
                    ✅ 当前状态：无紧急风险提示
                </div>
                """,
                    session_id,
                    None,
                    last_video,
                    latest,
                    all_sents,
                    gr.update(),
                )

            updated_history, risk_text, progress, updated_session, media_value = user_step(
                current_sentence, history, session_id
            )

            progress_text = _format_progress_heading(progress)

            if "高风险" in risk_text:
                risk_display = f"""
                <div class="risk-alert risk-high">
                    ⚠️ {risk_text}
                </div>
                """
            else:
                risk_display = f"""
                <div class="risk-alert risk-low">
                    ✅ {risk_text}
                </div>
                """
            
            # 🎯 关键修改：始终保留上一个视频，除非有明确的新视频
            if media_value is None:
                # 没有新视频时，保持显示上一个视频
                media_value = last_video
            else:
                # 有新视频时，更新状态
                last_video = media_value
            
            latest, all_sents = get_current_sentences()

            return (
                updated_history,
                risk_display,
                updated_session,
                media_value,
                last_video,
                latest,
                all_sents,
                gr.update(value=progress_text),
            )

        submit_sentence_btn.click(
            fn=submit_current_sentence_sync,
            inputs=[chatbot, session_state, last_video_state],
            outputs=[
                chatbot,
                risk_alert,
                session_state,
                video_sys,
                last_video_state,
                latest_sentence,
                all_sentences,
                progress_title,
            ]
        )

        report_button.click(
            lambda sid: _generate_report(sid),
            inputs=[session_state],
            outputs=[report_status],
        )

        # 修改初始化对话函数 - 修复视频闪屏问题
        def initialize_with_progress(session_id: str):
            history, sid, risk_text, progress, media_value = initialize_conversation(session_id)

            progress_text = _format_progress_heading(progress)

            if "高风险" in risk_text:
                risk_display = f"""
                <div class="risk-alert risk-high">
                    ⚠️ {risk_text}
                </div>
                """
            else:
                risk_display = f"""
                <div class="risk-alert risk-low">
                    ✅ {risk_text}
                </div>
                """

            return (
                history,
                sid,
                risk_display,
                media_value,
                media_value,  # 同时设置 last_video_state
                gr.update(value=progress_text),
            )

        demo.load(
            fn=initialize_with_progress,
            inputs=[session_state],
            outputs=[chatbot, session_state, risk_alert, video_sys, last_video_state, progress_title],
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

        # 简化的JavaScript - 只处理轮询
        demo.load(
            None,
            None,
            None,
            js="""
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
            """
        )

    return demo

def main():
    """直接启动Gradio应用（独立运行）"""
    print("🚀 启动智能心境健康评估系统...")
    print(f"📁 项目根目录: {PROJECT_ROOT}")
    print(f"🔑 听悟 AppKey: {settings.TINGWU_APPKEY or settings.ALIBABA_TINGWU_APPKEY or '未配置'}")
    
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )

if __name__ == "__main__":
    main()