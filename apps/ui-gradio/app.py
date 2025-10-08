from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
import requests
import websockets

from services.audio.tingwu_async_client import (
    create_realtime_task,
    stop_realtime_task,
)

API_BASE = (
    os.getenv("DM_API_BASE", os.getenv("API_BASE_URL", "http://localhost:8080"))
    or "http://localhost:8080"
).rstrip("/")


TARGET_SAMPLE_RATE = 16_000
FRAME_DURATION_SECONDS = 0.02  # 20 ms per frame
PCM_BYTES_PER_SAMPLE = 2  # int16 mono
FRAME_BYTE_LENGTH = int(TARGET_SAMPLE_RATE * FRAME_DURATION_SECONDS) * PCM_BYTES_PER_SAMPLE
CREATE_TASK_MAX_RETRIES = 3
CONNECT_MAX_RETRIES = 3
CONNECT_BACKOFF_SECONDS = 1.5
PING_INTERVAL_SECONDS = 15


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


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def _resample_audio(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == TARGET_SAMPLE_RATE:
        return audio
    if source_rate <= 0 or audio.size == 0:
        return np.asarray([], dtype=np.float32)

    duration = audio.shape[0] / float(source_rate)
    target_length = max(int(round(duration * TARGET_SAMPLE_RATE)), 1)
    source_positions = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, audio)
    return resampled.astype(np.float32)


def _to_pcm(audio: np.ndarray) -> bytes:
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * np.iinfo(np.int16).max).astype("<i2")
    return pcm.tobytes()


def _iter_frames(pcm_bytes: bytes) -> List[bytes]:
    if not pcm_bytes:
        return []
    frames: List[bytes] = []
    for offset in range(0, len(pcm_bytes), FRAME_BYTE_LENGTH):
        frame = pcm_bytes[offset : offset + FRAME_BYTE_LENGTH]
        if len(frame) < FRAME_BYTE_LENGTH:
            frame = frame + b"\x00" * (FRAME_BYTE_LENGTH - len(frame))
        frames.append(frame)
    return frames


def _extract_text(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if isinstance(result, str) and result.strip():
        return result.strip()
    if isinstance(result, dict):
        text = result.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    if isinstance(payload.get("text"), str) and payload["text"].strip():
        return payload["text"].strip()
    return None


async def _stream_realtime_audio(
    sample_rate: int,
    audio_data: np.ndarray,
    queue: "asyncio.Queue[object]",
) -> None:
    audio_data = audio_data.astype(np.float32)
    mono_audio = _ensure_mono(audio_data)
    resampled = _resample_audio(mono_audio, sample_rate)
    if resampled.size == 0:
        await queue.put("⚠️ 未检测到有效的音频样本。")
        return

    pcm_bytes = _to_pcm(resampled)
    frames = _iter_frames(pcm_bytes)
    await queue.put(
        "🎙️ 音频处理完成，准备推流："
        f"{len(resampled)} 个样本，拆分为 {len(frames)} 帧。"
    )

    task_id: Optional[str] = None
    ws_url: Optional[str] = None
    for attempt in range(1, CREATE_TASK_MAX_RETRIES + 1):
        try:
            ws_url, task_id = await create_realtime_task()
            await queue.put(f"✅ 已创建实时任务，第 {attempt} 次尝试成功。")
            break
        except Exception as exc:  # pragma: no cover - network failure path
            await queue.put(
                f"❌ 创建实时任务失败（第 {attempt} 次）：{exc}"
            )
            if attempt == CREATE_TASK_MAX_RETRIES:
                return
            await asyncio.sleep(attempt * 0.5)

    if not ws_url or not task_id:
        await queue.put("❌ 未能获得有效的 WebSocket 地址或任务 ID。")
        return

    async def _cleanup_task() -> None:
        if not task_id:
            return
        with contextlib.suppress(Exception):
            await stop_realtime_task(task_id)

    try:
        backoff = CONNECT_BACKOFF_SECONDS
        for attempt in range(1, CONNECT_MAX_RETRIES + 1):
            try:
                await queue.put(
                    f"🔌 正在连接 WebSocket（尝试 {attempt}/{CONNECT_MAX_RETRIES}）…"
                )
                async with websockets.connect(
                    ws_url,
                    ping_interval=None,
                    close_timeout=5,
                ) as ws:
                    await queue.put("✅ WebSocket 连接成功，开始推流。")

                    async def send_loop() -> None:
                        try:
                            for index, frame in enumerate(frames, start=1):
                                await ws.send(frame)
                                if index % 10 == 0 or index == len(frames):
                                    await queue.put(
                                        f"📤 已发送 {index}/{len(frames)} 帧。"
                                    )
                                await asyncio.sleep(FRAME_DURATION_SECONDS)
                            await queue.put("🛑 所有音频帧已发送，发送停止指令。")
                            await ws.send(json.dumps({"action": "Stop"}))
                        except Exception as exc:  # pragma: no cover
                            await queue.put(f"⚠️ 发送音频数据异常：{exc}")
                            raise

                    async def recv_loop() -> None:
                        try:
                            async for raw_msg in ws:
                                if isinstance(raw_msg, bytes):
                                    continue
                                try:
                                    message = json.loads(raw_msg)
                                except json.JSONDecodeError:
                                    await queue.put(
                                        "⚠️ 收到无法解析的消息，已忽略。"
                                    )
                                    continue
                                header = message.get("header", {})
                                payload = message.get("payload", {})
                                name = header.get("name")
                                if isinstance(name, str):
                                    await queue.put(f"📥 收到事件：{name}")
                                text = _extract_text(payload)
                                if text:
                                    await queue.put(f"📝 实时识别：{text}")
                        except websockets.ConnectionClosedOK:
                            await queue.put("🔚 WebSocket 正常关闭。")
                        except Exception as exc:  # pragma: no cover
                            await queue.put(f"⚠️ 接收识别结果异常：{exc}")
                            raise

                    async def heartbeat_loop() -> None:
                        try:
                            while True:
                                await asyncio.sleep(PING_INTERVAL_SECONDS)
                                await ws.ping()
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            return

                    send_task = asyncio.create_task(send_loop())
                    recv_task = asyncio.create_task(recv_loop())
                    heartbeat_task = asyncio.create_task(heartbeat_loop())

                    done, pending = await asyncio.wait(
                        {send_task, recv_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    heartbeat_task.cancel()
                    with contextlib.suppress(Exception):
                        await asyncio.gather(*pending, return_exceptions=True)
                    with contextlib.suppress(Exception):
                        await heartbeat_task
                    for task in done:
                        exc = task.exception()
                        if exc:
                            raise exc
                    break
            except Exception as exc:  # pragma: no cover - connection failure
                await queue.put(f"❌ WebSocket 连接失败：{exc}")
                if attempt == CONNECT_MAX_RETRIES:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2
    finally:
        with contextlib.suppress(Exception):
            await _cleanup_task()
        await queue.put("✅ 实时任务已结束。")


async def realtime_stream_to_frontend(
    audio: Optional[Tuple[int, np.ndarray]]
) -> AsyncGenerator[str, None]:
    if not audio:
        yield "⚠️ 未接收到麦克风音频。"
        return

    sample_rate, data = audio
    if data is None or getattr(data, "size", 0) == 0:
        yield "⚠️ 音频数据为空。"
        return

    queue: "asyncio.Queue[object]" = asyncio.Queue()
    sentinel = object()

    async def runner() -> None:
        try:
            await _stream_realtime_audio(sample_rate, data, queue)
        finally:
            await queue.put(sentinel)

    worker = asyncio.create_task(runner())
    try:
        while True:
            message = await queue.get()
            if message is sentinel:
                break
            if isinstance(message, str):
                yield message
    finally:
        if not worker.done():
            worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker


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
                gr.Markdown("## 🎧 实时语音识别")
                mic = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label="点击开始录音以推流至听悟",
                )
                realtime_output = gr.Textbox(
                    label="实时识别输出",
                    lines=8,
                    interactive=False,
                )
                mic.stream(
                    fn=realtime_stream_to_frontend,
                    inputs=mic,
                    outputs=realtime_output,
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
        ) -> Tuple[List[Tuple[str, str]], str, Optional[str], str, str, Dict[str, Any], Optional[str]]:
            chat, risk_text, progress, sid, audio_value = user_step(
                message, audio_path, history, session_id
            )
            return chat, "", None, sid, risk_text, progress, audio_value

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

        report_button.click(
            lambda sid: _generate_report(sid),
            inputs=[session_state],
            outputs=[report_status],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
