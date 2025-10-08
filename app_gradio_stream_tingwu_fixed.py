#!/usr/bin/env python
"""Gradio → Tingwu realtime streaming demo with continuous captions.

This module mounts a Gradio interface that records microphone audio,
resamples it to 16 kHz mono, and streams PCM frames to the Tingwu
realtime WebSocket interface while yielding every status / transcript
message back to the UI.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Awaitable
from contextlib import suppress
from typing import AsyncGenerator, Optional, Sequence

import gradio as gr
import numpy as np
import websockets
from services.audio.tingwu_client import create_realtime_task, stop_realtime_task


TARGET_SAMPLE_RATE = 16_000
FRAME_INTERVAL_SECONDS = 0.04  # 40 ms per requirement
SAMPLES_PER_FRAME = int(TARGET_SAMPLE_RATE * FRAME_INTERVAL_SECONDS)
PCM_BYTES_PER_SAMPLE = 2  # int16
FRAME_BYTE_LENGTH = SAMPLES_PER_FRAME * PCM_BYTES_PER_SAMPLE


try:  # pragma: no cover - compatibility shim when running on Python 3.10
    from asyncio import TaskGroup
except ImportError:  # Python < 3.11 fallback

    class TaskGroup:  # type: ignore[override]
        """Minimal TaskGroup shim for Python 3.10 environments."""

        def __init__(self) -> None:
            self._tasks: list[asyncio.Task[object]] = []

        async def __aenter__(self) -> "TaskGroup":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:
            try:
                if exc is not None:
                    for task in self._tasks:
                        task.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)
            finally:
                self._tasks.clear()
            return None

        def create_task(self, coro: Awaitable[object]) -> asyncio.Task:
            task = asyncio.create_task(coro)
            self._tasks.append(task)
            return task


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def _resample_to_target(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == TARGET_SAMPLE_RATE:
        return audio
    if source_rate <= 0 or audio.size == 0:
        return np.asarray([], dtype=np.float32)

    duration = audio.shape[0] / float(source_rate)
    new_length = int(math.floor(duration * TARGET_SAMPLE_RATE))
    if new_length <= 0:
        return np.asarray([], dtype=np.float32)

    time_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    time_new = np.linspace(0.0, duration, num=new_length, endpoint=False)
    resampled = np.interp(time_new, time_old, audio)
    return resampled.astype(np.float32)


def _to_pcm_frames(audio: np.ndarray) -> bytes:
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * np.iinfo(np.int16).max).astype("<i2")
    return pcm.tobytes()


def _iter_pcm_frames(pcm_bytes: bytes) -> Sequence[bytes]:
    if not pcm_bytes:
        return []
    frames: list[bytes] = []
    for offset in range(0, len(pcm_bytes), FRAME_BYTE_LENGTH):
        frame = pcm_bytes[offset : offset + FRAME_BYTE_LENGTH]
        if len(frame) < FRAME_BYTE_LENGTH:
            frame = frame + b"\x00" * (FRAME_BYTE_LENGTH - len(frame))
        frames.append(frame)
    return frames


async def stream_audio_to_tingwu(
    audio: np.ndarray,
    source_rate: int,
    queue: "asyncio.Queue[str]",
) -> None:
    if audio is None or audio.size == 0:
        await queue.put("未检测到有效音频帧")
        return

    mono_audio = _ensure_mono(audio.astype(np.float32))
    resampled = _resample_to_target(mono_audio, source_rate)
    if resampled.size == 0:
        await queue.put("未检测到有效音频帧")
        return

    pcm_bytes = _to_pcm_frames(resampled)
    frames = _iter_pcm_frames(pcm_bytes)
    await queue.put(
        f"🎚️ 音频处理完成：{len(resampled)} 帧样本，共 {len(frames)} 个分片"
    )

    loop = asyncio.get_running_loop()
    try:
        ws_url, task_id = await loop.run_in_executor(None, create_realtime_task)
    except Exception as exc:  # pragma: no cover - network failure path
        await queue.put(f"❌ 创建实时任务失败：{exc}")
        return

    await queue.put("🔌 Tingwu 任务已创建，准备建立 WebSocket 连接…")

    async def _close_task() -> None:
        with suppress(Exception):
            await loop.run_in_executor(None, stop_realtime_task, task_id)

    try:
        async with websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as ws:
            await queue.put("✅ WebSocket 已连接，开始实时推流…")

            async def send_loop() -> None:
                try:
                    for idx, frame in enumerate(frames, start=1):
                        await ws.send(frame)
                        await queue.put(
                            f"📤 发送第 {idx} 帧（{len(frame)} 字节）"
                        )
                        await asyncio.sleep(FRAME_INTERVAL_SECONDS)
                    await queue.put("🛑 音频帧发送完毕，发送结束指令…")
                    await ws.send(json.dumps({"action": "Stop"}))
                except Exception as exc:  # pragma: no cover - runtime failure
                    await queue.put(f"❌ 发送音频时出现异常：{exc}")
                    raise

            async def recv_loop() -> None:
                try:
                    async for raw_msg in ws:
                        await queue.put(f"📥 收到原始消息: {raw_msg}")
                        with suppress(json.JSONDecodeError, TypeError):
                            payload = json.loads(raw_msg)
                            text = (
                                payload.get("text")
                                or payload.get("result")
                                or payload.get("payload", {}).get("text")
                                or payload.get("payload", {}).get("result")
                            )
                            if text:
                                await queue.put(f"📝 实时识别：{text}")
                except websockets.ConnectionClosedOK:
                    await queue.put("🔚 WebSocket 正常关闭。")
                except Exception as exc:  # pragma: no cover - runtime failure
                    await queue.put(f"⚠️ 接收识别结果时异常：{exc}")
                    raise

            async with TaskGroup() as tg:
                tg.create_task(send_loop())
                tg.create_task(recv_loop())

    except Exception as exc:  # pragma: no cover - connection failures
        await queue.put(f"❌ WebSocket 连接失败：{exc}")
    finally:
        await _close_task()
        await queue.put("✅ Tingwu 任务已结束。")


async def _gradio_stream(audio) -> AsyncGenerator[str, None]:
    if audio is None:
        yield "未检测到有效音频帧"
        return

    if isinstance(audio, tuple):
        sample_rate, data = audio
    elif isinstance(audio, dict):
        sample_rate = audio.get("sample_rate", TARGET_SAMPLE_RATE)
        data = audio.get("data")
    else:
        sample_rate, data = TARGET_SAMPLE_RATE, audio

    if data is None:
        yield "未检测到有效音频帧"
        return

    np_data = np.asarray(data, dtype=np.float32)
    queue: "asyncio.Queue[str]" = asyncio.Queue()

    async def runner() -> None:
        await stream_audio_to_tingwu(np_data, int(sample_rate), queue)
        await queue.put("流程结束")

    worker = asyncio.create_task(runner())
    try:
        while True:
            message = await queue.get()
            yield message
            if message.endswith("流程结束"):
                break
    finally:
        if not worker.done():
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker


with gr.Blocks() as demo:
    gr.Markdown("## 🎧 听悟实时语音识别（16 kHz 单声道）")
    audio_in = gr.Audio(
        sources=["microphone"],
        type="numpy",
        streaming=True,
        format="wav",
        label="🎙️ 麦克风输入",
    )
    transcript = gr.Textbox(label="实时识别输出", lines=12)
    audio_in.stream(fn=_gradio_stream, inputs=audio_in, outputs=transcript)

    gr.Markdown("---")
    gr.Markdown("### 🎙️ 录音上传识别（自动调用 Tingwu 客户端）")
    recorder = gr.Audio(
        sources=["microphone", "upload"],
        type="numpy",
        streaming=False,
        format="wav",
        label="录音或上传音频（16 kHz 单声道）",
    )

    def _append_transcript(text: str | None, addition: str) -> str:
        base = text or ""
        if base and not base.endswith("\n"):
            base += "\n"
        return base + addition

    def _process_recording(audio, current_text: str | None) -> str:
        if audio is None:
            return _append_transcript(current_text, "❌ 未检测到音频输入或上传失败")

        if isinstance(audio, tuple):
            sample_rate, data = audio
        elif isinstance(audio, dict):
            sample_rate = audio.get("sample_rate", TARGET_SAMPLE_RATE)
            data = audio.get("data")
        else:
            sample_rate, data = TARGET_SAMPLE_RATE, audio

        sample_rate = int(sample_rate) if sample_rate else TARGET_SAMPLE_RATE

        if data is None:
            return _append_transcript(current_text, "❌ 未检测到音频输入或上传失败")

        np_data = np.asarray(data, dtype=np.float32)
        if np_data.size == 0:
            return _append_transcript(current_text, "❌ 未检测到音频输入或上传失败")

        mono_audio = _ensure_mono(np_data)
        resampled = _resample_to_target(mono_audio, sample_rate)
        if resampled.size == 0:
            return _append_transcript(current_text, "❌ 未检测到音频输入或上传失败")

        pcm_bytes = _to_pcm_frames(resampled)
        if not pcm_bytes:
            return _append_transcript(current_text, "❌ 未检测到音频输入或上传失败")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pcm") as tmp:
                tmp.write(pcm_bytes)
                temp_path = tmp.name

            try:
                result = subprocess.run(
                    ["python", "-m", "services.audio.tingwu_client", temp_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                stdout = result.stdout.strip()
                addition = stdout or "✅ Tingwu 客户端执行完成（无输出）"
                return _append_transcript(current_text, addition)
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.strip() if exc.stderr else str(exc)
                return _append_transcript(current_text, f"❌ 调用失败：{stderr}")
            except Exception as exc:  # pragma: no cover - unexpected failure
                return _append_transcript(current_text, f"❌ 调用失败：{exc}")
            finally:
                with suppress(Exception):
                    os.remove(temp_path)
        except Exception as exc:  # pragma: no cover - temp file failure
            return _append_transcript(current_text, f"❌ 音频处理异常：{exc}")

    recorder.change(
        fn=_process_recording,
        inputs=[recorder, transcript],
        outputs=transcript,
    )


async def _launch() -> None:
    await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: demo.queue().launch(server_name="0.0.0.0", server_port=8001),
    )


if __name__ == "__main__":
    asyncio.run(_launch())
