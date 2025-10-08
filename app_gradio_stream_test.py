"""Gradio 页面：16 kHz 录音 → 听悟 WebSocket 推流 → 实时字幕."""

from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import gradio as gr
import numpy as np
import websockets

from packages.common.config import settings
from services.audio.tingwu_async_client import (
    create_realtime_task,
    stop_realtime_task,
)


@dataclass(slots=True)
class TingwuStreamConfig:
    """Normalized Tingwu realtime streaming configuration."""

    appkey: str
    format: str
    language: str
    sample_rate: int
    frame_ms: int = 10

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
        sample_rate=16000,
    )


def _ensure_mono(audio_data: np.ndarray) -> np.ndarray:
    if audio_data.ndim == 1:
        return audio_data
    return np.mean(audio_data, axis=1)


def _resample_if_needed(audio_data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return audio_data
    if audio_data.size == 0:
        return audio_data
    original_indices = np.arange(audio_data.shape[0], dtype=np.float32)
    target_length = max(int(round(audio_data.shape[0] * target_sr / original_sr)), 1)
    target_indices = np.linspace(0, original_indices[-1], num=target_length, dtype=np.float32)
    return np.interp(target_indices, original_indices, audio_data).astype(np.float32)


def _pcm_bytes(audio_data: np.ndarray) -> bytes:
    normalized = np.clip(audio_data, -1.0, 1.0)
    return (normalized * 32767).astype("<i2").tobytes()


async def stream_audio_to_tingwu(
    audio_data: np.ndarray, sample_rate: int, queue: asyncio.Queue[str]
) -> None:
    """推送音频到 Tingwu 并实时接收识别结果."""

    cfg = _build_config()
    mono_audio = _ensure_mono(audio_data.astype(np.float32))
    processed_audio = _resample_if_needed(mono_audio, sample_rate, cfg.sample_rate)
    pcm_bytes = _pcm_bytes(processed_audio)

    task_id: Optional[str] = None

    try:
        ws_url, task_id = await create_realtime_task()
    except Exception as exc:  # pylint: disable=broad-except
        await queue.put(f"❌ 创建听悟实时任务失败: {exc}")
        return

    print(f"✅ 任务已创建 ({task_id})", flush=True)
    await queue.put(f"✅ 任务已创建 ({task_id})")

    start_message = {
        "header": {
            "namespace": "SpeechTranscriber",
            "name": "StartTranscription",
            "appkey": cfg.appkey,
        },
        "payload": {
            "format": cfg.format,
            "sample_rate": cfg.sample_rate,
            "language": cfg.language,
            "enable_punctuation_prediction": True,
            "enable_inverse_text_normalization": True,
            "enable_semantic_sentence_detection": True,
        },
    }
    stop_message = {
        "header": {
            "namespace": "SpeechTranscriber",
            "name": "StopTranscription",
        },
        "payload": {},
    }

    def _extract_result_text(payload: dict) -> Optional[str]:
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, str):
            return result.strip() or None
        if isinstance(result, dict):
            text = result.get("text") if isinstance(result.get("text"), str) else None
            if text:
                return text.strip() or None
        if isinstance(result, list):
            texts = []
            for item in result:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
                elif isinstance(item, str) and item.strip():
                    texts.append(item.strip())
            if texts:
                return " ".join(texts)
        return None

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
                name = header.get("name")
                if name:
                    print(f"📥 收到消息 {name}", flush=True)
                    if isinstance(payload, dict) and name in {
                        "SentenceBegin",
                        "SentenceEnd",
                        "TranscriptionCompleted",
                    }:
                        result_text = _extract_result_text(payload)
                        log_payload = result_text or json.dumps(
                            payload, ensure_ascii=False
                        )
                        log = f"🟢 {name}: {log_payload}"
                        print(log, flush=True)
                        await queue.put(log)
                status = header.get("status")
                if isinstance(status, int) and status >= 40000000:
                    detail = payload.get("message") or payload.get("error_message")
                    await queue.put(
                        f"⚠️ 听悟返回错误({status}): {detail or json.dumps(payload, ensure_ascii=False)}"
                    )
                    continue
                text = None
                if isinstance(payload, dict):
                    result_text = _extract_result_text(payload)
                    if result_text:
                        text = result_text
                    elif isinstance(payload.get("text"), str) and payload["text"].strip():
                        text = payload["text"].strip()
                if text:
                    await queue.put(f"📝 实时识别结果：{text}")
        except Exception as exc:  # pylint: disable=broad-except
            await queue.put(f"⚠️ WebSocket 监听中断: {exc}")

    async def _run_with_connection(connect_kwargs: dict) -> None:
        async with websockets.connect(ws_url, **connect_kwargs) as ws:
            print("✅ WebSocket 已连接", flush=True)
            await queue.put("✅ WebSocket 已连接")
            await ws.send(json.dumps(start_message, ensure_ascii=False))

            receiver = asyncio.create_task(receive_results(ws))
            frame_size = 640
            total_frames = (len(pcm_bytes) + frame_size - 1) // frame_size
            try:
                await asyncio.sleep(0.2)
                for frame_index in range(total_frames):
                    start = frame_index * frame_size
                    await ws.send(pcm_bytes[start : start + frame_size])
                    print(f"📤 已发送第 {frame_index + 1} 帧", flush=True)
                    if (frame_index + 1) % 100 == 0:
                        print(
                            f"📤 已累计发送 {frame_index + 1} 帧 (~{(frame_index + 1) * 10} ms)",
                            flush=True,
                        )
                    await asyncio.sleep(0.01)

                await ws.send(json.dumps(stop_message, ensure_ascii=False))
                print("🔚 Stop 指令已发送，等待 3 秒后关闭", flush=True)
                await asyncio.sleep(3)
                print("🔚 Stop 完成并关闭 WebSocket", flush=True)
                await queue.put("🔚 Stop 完成并关闭 WebSocket")
            finally:
                if not receiver.done():
                    receiver.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await receiver

    connect_kwargs = {"ping_interval": 10}

    try:
        try:
            await _run_with_connection(connect_kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            if "302" in str(exc):
                connect_kwargs["ssl"] = ssl._create_unverified_context()
                await _run_with_connection(connect_kwargs)
            else:
                raise
    except Exception as exc:  # pylint: disable=broad-except
        await queue.put(f"❌ 推流失败: {exc}")
    finally:
        if task_id:
            try:
                await stop_realtime_task(task_id)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"⚠️ 停止实时任务失败: {exc}", flush=True)


async def start_realtime_stream(audio: Optional[tuple[int, np.ndarray]]) -> AsyncGenerator[str, None]:
    """录音回调，实时显示识别结果."""

    if audio is None:
        yield "未检测到音频输入"
        return

    sample_rate, data = audio
    queue: "asyncio.Queue[str]" = asyncio.Queue()

    async def run_stream() -> None:
        try:
            await stream_audio_to_tingwu(data, sample_rate, queue)
        finally:
            await queue.put("✅ 流程结束")

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
