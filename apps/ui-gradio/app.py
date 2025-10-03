from __future__ import annotations

import uuid
from typing import Any, Dict, List, Tuple

import gradio as gr

from services.audio import asr_adapter
from services.orchestrator.langgraph_min import orchestrator


def _init_session() -> str:
    return str(uuid.uuid4())


def user_step(message: str, history: List[Tuple[str, str]], session_id: str) -> Tuple[List[Tuple[str, str]], str, Dict[str, Any]]:
    segments = asr_adapter.transcribe(text=message)
    result = orchestrator.step(session_id, segments=segments)
    history = history + [(message, result.get("next_utterance", ""))]
    progress = result.get("progress", {})
    risk_info = result.get("risk")
    risk_text = "⚠️ 检测到高风险，请立即寻求紧急帮助。" if risk_info else "无紧急风险提示。"
    return history, risk_text, progress


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Depression Agent UI") as demo:
        session_state = gr.State(_init_session())

        gr.Markdown("# 抑郁随访助手")
        chatbot = gr.Chatbot(height=400, label="对话")
        text_input = gr.Textbox(label="患者输入", placeholder="请输入文本（语音功能即将上线）")
        gr.Textbox(label="语音录制占位", value="未来将接入音频录制模块", interactive=False)
        risk_alert = gr.Markdown("无紧急风险提示。")
        progress_display = gr.JSON(label="进度状态")

        def _on_submit(message: str, history: List[Tuple[str, str]], session_id: str):
            chat, risk_text, progress = user_step(message, history, session_id)
            return chat, "", session_id, risk_text, progress

        text_input.submit(
            _on_submit,
            inputs=[text_input, chatbot, session_state],
            outputs=[chatbot, text_input, session_state, risk_alert, progress_display],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
