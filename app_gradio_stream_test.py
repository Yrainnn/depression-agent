from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import uvicorn
import asyncio

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/stream")
async def audio_stream(ws: WebSocket):
    await ws.accept()
    print("✅ WebSocket connected.")
    while True:
        try:
            data = await ws.receive_bytes()
            # 这里 data 就是 PCM 分片，可直接送听悟 SDK
            print(f"🎧 Received audio chunk: {len(data)} bytes")
        except Exception as e:
            print("❌ Connection closed:", e)
            break


# === Gradio 部分 ===
def record_and_send(audio):
    # audio 是 (sample_rate, np.ndarray)
    if audio is None:
        return "未检测到音频输入"
    sample_rate, data = audio
    pcm_bytes = (data * 32767).astype("<i2").tobytes()

    import websockets

    async def send_ws():
        async with websockets.connect("ws://127.0.0.1:8080/stream") as ws:
            await ws.send(pcm_bytes)
            await asyncio.sleep(0.1)
            await ws.close()

    asyncio.run(send_ws())
    return f"采样率：{sample_rate} Hz，音频长度：{len(pcm_bytes)} 字节"


with gr.Blocks() as demo:
    gr.Markdown("## 🎤 Gradio 实时录音推流测试")
    mic = gr.Audio(source="microphone", type="numpy")
    output = gr.Textbox()
    mic.change(fn=record_and_send, inputs=[mic], outputs=[output])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
