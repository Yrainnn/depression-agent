"""Gradio 实时录音推流 Tingwu WebSocket 示例."""
import asyncio
import contextlib
import json
from typing import AsyncGenerator, Optional

import gradio as gr
import numpy as np
import websockets

# ===== Tingwu 实时识别配置 =====
# TODO: 将以下常量替换为真实值
TINGWU_WS_URL = "wss://your-tingwu-endpoint"
API_KEY = "your-tingwu-api-key"


async def stream_audio_to_tingwu(
    audio_data: np.ndarray, sample_rate: int, queue: asyncio.Queue[str]
) -> None:
    """推送音频到 Tingwu 并实时接收识别结果."""
    # ---- 音频预处理：单声道 + 16 kHz ----
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    if sample_rate != 16000:
        await queue.put(f"⚠️ 检测到采样率 {sample_rate}Hz，请确保为 16 kHz")

    pcm_bytes = (audio_data * 32767).astype(np.int16).tobytes()
    chunk_size = 3200  # ≈100 ms

    try:
        async with websockets.connect(
            TINGWU_WS_URL,
            extra_headers={"Authorization": f"Bearer {API_KEY}"},
            ping_interval=20,
        ) as ws:
            await queue.put("✅ WebSocket 已连接，开始推流...")

            # ---- 异步接收识别结果 ----
            async def receive_results() -> None:
                try:
                    async for msg in ws:
                        data = json.loads(msg)
                        if "text" in data:
                            await queue.put(f"实时识别: {data['text']}")
                        elif "result" in data:
                            await queue.put(f"最终结果: {data['result']}")
                except Exception as exc:  # pylint: disable=broad-except
                    await queue.put(f"⚠️ WebSocket 关闭: {exc}")

            receiver = asyncio.create_task(receive_results())

            # ---- 音频推流 ----
            for idx in range(0, len(pcm_bytes), chunk_size):
                await ws.send(pcm_bytes[idx : idx + chunk_size])
                await asyncio.sleep(0.1)

            await ws.send(json.dumps({"end": True}))
            await queue.put("🛑 推流完成，等待识别结束...")
            await receiver  # 等待接收任务结束
    except Exception as exc:  # pylint: disable=broad-except
        await queue.put(f"❌ 推流失败: {exc}")


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
                if "流程结束" in text or "关闭" in text or "失败" in text:
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
