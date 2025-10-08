from __future__ import annotations

import audioop
import contextlib
import os
import shutil
import subprocess
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests

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


def _convert_audio_to_pcm_16k(audio_path: str) -> str:
    if not audio_path:
        raise ValueError("❌ 未检测到音频输入或上传失败")

    source_path = Path(audio_path)
    if not source_path.exists():
        raise FileNotFoundError("❌ 未检测到音频输入或上传失败")

    tmp_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pcm")
    tmp_handle.close()
    tmp_path = tmp_handle.name

    try:
        if source_path.suffix.lower() == ".pcm":
            with open(source_path, "rb") as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return tmp_path

        if source_path.suffix.lower() != ".wav":
            raise ValueError("仅支持 .wav 或 .pcm 格式音频")

        with contextlib.closing(wave.open(str(source_path), "rb")) as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        if channels <= 0:
            raise ValueError("音频通道数无效")

        if sample_width != 2:
            frames = audioop.lin2lin(frames, sample_width, 2)

        if channels > 1:
            frames = audioop.tomono(frames, 2, 1, 1)

        if frame_rate != 16000:
            frames, _ = audioop.ratecv(frames, 2, 1, frame_rate, 16000, None)

        with open(tmp_path, "wb") as pcm_file:
            pcm_file.write(frames)

        return tmp_path
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def _run_tingwu_client(pcm_path: str) -> str:
    command = ["python", "-m", "services.audio.tingwu_client", pcm_path]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as exc:  # noqa: TRY003 - surface to UI
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        message_parts = ["❌ 听悟转写失败："]
        if stdout:
            message_parts.append(stdout)
        if stderr:
            message_parts.append(stderr)
        return "\n".join(message_parts)

    output_text = (completed.stdout or "").strip()
    if not output_text:
        return "✅ 听悟转写完成（无返回内容）。"
    return output_text


def _append_transcript(existing: str, new_message: str) -> str:
    existing = existing.strip()
    if not existing:
        return new_message
    return f"{existing}\n{new_message}"


def _handle_tingwu_audio(
    audio_path: Optional[str], transcript: str
) -> str:
    transcript = transcript or ""
    if not audio_path:
        return _append_transcript(transcript, "❌ 未检测到音频输入或上传失败")

    try:
        pcm_path = _convert_audio_to_pcm_16k(audio_path)
        try:
            result_text = _run_tingwu_client(pcm_path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(pcm_path)
    except Exception as exc:  # noqa: BLE001 - surface到UI
        return _append_transcript(transcript, f"❌ 音频处理失败：{exc}")

    return _append_transcript(transcript, result_text)


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


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Depression Agent UI") as demo:
        session_state = gr.State(_init_session())

        gr.Markdown("# 抑郁随访助手")

        with gr.Tabs():
            with gr.Tab("评估"):
                chatbot = gr.Chatbot(height=400, label="对话")
                text_input = gr.Textbox(label="患者输入", placeholder="请输入文本")
                audio_input = gr.Audio(
                    label="录音/上传音频 (16k 单声道)",
                    sources=["microphone", "upload"],
                    type="filepath",
                )
                tingwu_transcript = gr.Textbox(
                    label="听悟识别结果",
                    value="",
                    lines=8,
                    interactive=False,
                )
                audio_sys = gr.Audio(label="系统语音", interactive=False, autoplay=True)
                risk_alert = gr.Markdown("无紧急风险提示。")
                progress_display = gr.JSON(label="进度状态")
                send_button = gr.Button("发送")

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

        audio_input.change(
            _handle_tingwu_audio,
            inputs=[audio_input, tingwu_transcript],
            outputs=[tingwu_transcript],
        )

        if hasattr(audio_input, "stop_recording"):
            audio_input.stop_recording(
                _handle_tingwu_audio,
                inputs=[audio_input, tingwu_transcript],
                outputs=[tingwu_transcript],
            )

        report_button.click(
            lambda sid: _generate_report(sid),
            inputs=[session_state],
            outputs=[report_status],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
