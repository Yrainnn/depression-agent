from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests


API_BASE = (
    os.getenv("DM_API_BASE", os.getenv("API_BASE_URL", "http://localhost:8080"))
    or "http://localhost:8080"
).rstrip("/")
import uuid
from typing import Any, Dict, List, Tuple

import gradio as gr

import os
import sys
# 获取项目根目录（根据目录结构，根目录是 app.py 上两级目录）
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# 把根目录加入 PYTHONPATH
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    
from services.audio import asr_adapter
from services.orchestrator.langgraph_min import orchestrator


def _init_session() -> str:
    return str(uuid.uuid4())


def _call_dm_step(sid: str, text: Optional[str] = None, audio_ref: Optional[str] = None) -> Dict[str, Any]:
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
        response = requests.post(url, files=files, data=data, timeout=30)
    response.raise_for_status()
    payload = response.json()
    audio_ref = payload.get("audio_ref")
    if not audio_ref:
        raise ValueError("audio_ref missing in upload response")
    return audio_ref


def user_step(
    message: str,
    audio_path: Optional[str],
    history: List[Tuple[str, str]],
    session_id: str,
) -> Tuple[List[Tuple[str, str]], str, Dict[str, Any], str]:
    message = message or ""
    text_payload = message.strip() or None
    audio_ref: Optional[str] = None
    risk_text = "无紧急风险提示。"
    progress: Dict[str, Any] = {}

    try:
        if audio_path:
            audio_ref = _upload_audio(session_id, audio_path)
            # When audio is provided, text becomes optional.
            if not text_payload:
                text_payload = None

        result = _call_dm_step(session_id, text=text_payload, audio_ref=audio_ref)
    except Exception as exc:  # noqa: BLE001 - surface API failures to the UI
        user_label: str
        if text_payload:
            user_label = message
        elif audio_path:
            user_label = f"[音频] {Path(audio_path).name}"
        else:
            user_label = "[空输入]"

        history = history + [(user_label, f"❌ 请求失败：{exc}")]
        return history, "⚠️ 请求失败，请稍后重试。", {}, session_id

    assistant_reply = result.get("next_utterance", "")
    previews = result.get("segments_previews") or []
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
        # First question fetch or empty input -> only append assistant reply.
        history = history + [(None, assistant_reply)]

    progress = result.get("progress", {})
    risk_flag = result.get("risk_flag", False)
    risk_text = (
        "⚠️ 检测到高风险，请立即寻求紧急帮助。" if risk_flag else "无紧急风险提示。"
    )

    return history, risk_text, progress, session_id


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Depression Agent UI") as demo:
        session_state = gr.State(_init_session())

        gr.Markdown("# 抑郁随访助手")
        chatbot = gr.Chatbot(height=400, label="对话")
        text_input = gr.Textbox(label="患者输入", placeholder="请输入文本")
        audio_input = gr.File(label="上传音频(16k mono)", type="filepath")
        risk_alert = gr.Markdown("无紧急风险提示。")
        progress_display = gr.JSON(label="进度状态")
        send_button = gr.Button("发送")

        def _on_submit(
            message: str,
            audio_path: Optional[str],
            history: List[Tuple[str, str]],
            session_id: str,
        ):
            chat, risk_text, progress, sid = user_step(message, audio_path, history, session_id)
            return chat, "", None, sid, risk_text, progress

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
            ],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
