"""Gradio 页面：16 kHz 录音 → 听悟 WebSocket 推流 → 实时字幕."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import gradio as gr
import numpy as np
import websockets

from packages.common.config import settings
from services.audio.tingwu_client import create_realtime_task, stop_realtime_task


@dataclass(slots=True)
class TingwuStreamConfig:
    """Normalized Tingwu realtime streaming configuration."""

    appkey: str
    format: str
    language: str
    sample_rate: int
    frame_ms: int = 40

    @property
    def frame_bytes(self) -> int:
        """Number of bytes per PCM frame according to configured frame size."""

        samples_per_frame = max(int(self.sample_rate * self.frame_ms / 1000), 1)
        return samples_per_frame * 2  # 16-bit mono PCM


def _build_config() -> TingwuStreamConfig:
    appkey = settings.TINGWU_APPKEY or settings.ALIBABA_TINGWU_APPKEY
    if not appkey:
        raise RuntimeError("请在 .env 中配置 TINGWU_APPKEY 或 ALIBABA_TINGWU_APPKEY")

    return TingwuStreamConfig(
        appkey=appkey,
        format=settings.TINGWU_FORMAT or "pcm",
        language=settings.TINGWU_LANG or "cn",
        sample_rate=settings.TINGWU_SAMPLE_RATE or 16000,
    )


def _ensure_mono(audio_data: np.ndarray) -> np.ndarray:
    if audio_data.ndim == 1:
        return audio_data
    return np.mean(audio_data, axis=1)


def _resample_if_needed(audio_data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return audio_data
    duration = audio_data.shape[0] / float(original_sr)
    target_samples = max(int(math.ceil(duration * target_sr)), 1)
    source_times = np.linspace(0.0, duration, num=audio_data.shape[0], endpoint=False)
    target_times = np.linspace(0.0, duration, num=target_samples, endpoint=False)
    return np.interp(target_times, source_times, audio_data).astype(np.float32)


def _pcm_bytes(audio_data: np.ndarray) -> bytes:
    normalized = np.clip(audio_data, -1.0, 1.0)
    return (normalized * 32767).astype("<i2").tobytes()


def _extract_text(payload: dict) -> Optional[str]:
    for key in ("display_text", "text", "result"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def stream_audio_to_tingwu(
    audio_data: np.ndarray, sample_rate: int, queue: asyncio.Queue[str]
) -> None:
    """推送音频到 Tingwu 并实时接收识别结果."""

    cfg = _build_config()
    mono_audio = _ensure_mono(audio_data.astype(np.float32))
    processed_audio = _resample_if_needed(mono_audio, sample_rate, cfg.sample_rate)
    pcm_bytes = _pcm_bytes(processed_audio)

    try:
        ws_url, task_id = await asyncio.to_thread(create_realtime_task)
    except Exception as exc:  # pylint: disable=broad-except
        await queue.put(f"❌ 创建听悟实时任务失败: {exc}")
        return

    start_message = {
        "header": {
            "namespace": "SpeechTranscription",
            "name": "StartTranscription",
            "appkey": cfg.appkey,
            "message_id": str(uuid.uuid4()),
            "task_id": task_id,
        },
        "payload": {
            "format": cfg.format,
            "sample_rate": cfg.sample_rate,
            "language": cfg.language,
            "enable_intermediate_result": True,
            "enable_punctuation_prediction": True,
            "enable_inverse_text_normalization": True,
            "enable_semantic_sentence_detection": True,
        },
    }
    stop_message = {
        "header": {
            "namespace": "SpeechTranscription",
            "name": "StopTranscription",
            "appkey": cfg.appkey,
            "message_id": str(uuid.uuid4()),
            "task_id": task_id,
        }
    }

    async def receive_results(ws):  # type: ignore[no-untyped-def]
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                header = message.get("header", {})
                payload = message.get("payload", {})
                status = header.get("status")
                if isinstance(status, int) and status >= 40000000:
                    detail = payload.get("message") or payload.get("error_message")
                    await queue.put(
                        f"⚠️ 听悟返回错误({status}): {detail or json.dumps(payload, ensure_ascii=False)}"
                    )
                    continue
                text = _extract_text(payload)
                if not text:
                    continue
                name = (header.get("name") or "").lower()
                prefix = "实时识别"
                if "sentenceend" in name or "completed" in name or "result" in name:
                    prefix = "最终结果"
                await queue.put(f"{prefix}: {text}")
        except Exception as exc:  # pylint: disable=broad-except
            await queue.put(f"⚠️ WebSocket 监听中断: {exc}")

    try:
        async with websockets.connect(ws_url, ping_interval=10) as ws:
            await queue.put(f"✅ 已连接听悟实时任务（TaskId: {task_id}）")
            await ws.send(json.dumps(start_message, ensure_ascii=False))

            receiver = asyncio.create_task(receive_results(ws))
            frame_bytes = max(cfg.frame_bytes, 640)
            for idx in range(0, len(pcm_bytes), frame_bytes):
                await ws.send(pcm_bytes[idx : idx + frame_bytes])
                await asyncio.sleep(cfg.frame_ms / 1000.0)

            await ws.send(json.dumps(stop_message, ensure_ascii=False))
            await queue.put("🛑 音频推流完成，等待识别收尾...")
            await receiver
    except Exception as exc:  # pylint: disable=broad-except
        await queue.put(f"❌ 推流失败: {exc}")
    finally:
        await asyncio.to_thread(stop_realtime_task, task_id)


def start_realtime_stream(audio: Optional[tuple[int, np.ndarray]]) -> AsyncGenerator[str, None]:
    """录音回调，实时显示识别结果."""

    if audio is None:

        async def no_audio() -> AsyncGenerator[str, None]:
            yield "未检测到音频输入"

        return no_audio()

    sample_rate, data = audio
    queue: "asyncio.Queue[str]" = asyncio.Queue()

    async def run_stream() -> None:
        await stream_audio_to_tingwu(data, sample_rate, queue)
        await queue.put("✅ 流程结束")

    async def update_output() -> AsyncGenerator[str, None]:
        task = asyncio.create_task(run_stream())
        try:
            while True:
                text = await queue.get()
                yield text
                if any(keyword in text for keyword in ("流程结束", "失败", "中断")):
                    break
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    return update_output()


with gr.Blocks() as demo:
    gr.Markdown("## 🎧 听悟实时语音识别 (16 kHz 单声道, 实时字幕)")
    mic = gr.Audio(
        sources=["microphone"],
        type="numpy",
        streaming=True,
        label="🎙️ 麦克风输入 (16 kHz 单声道)",
    )
    output = gr.Textbox(label="实时识别结果", lines=6)
    mic.stream(fn=start_realtime_stream, inputs=mic, outputs=output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8001)
