#!/usr/bin/env python
# coding=utf-8
"""
services/audio/tingwu_client.py
-------------------------------
通义听悟实时语音识别（文件回放式）封装为项目内部模块：
- 自动创建实时任务（OpenAPI）
- 使用 NLS SDK 建立实时会话，分帧推流 .wav/.pcm
- 收集 SentenceEnd 文本
- 停止任务并返回最终文本
"""
import json
import sys
import threading
import time

import nls
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

from packages.common.config import settings

nls.enableTrace(False)

REGION = settings.TINGWU_REGION or "cn-beijing"
DOMAIN = f"tingwu.{REGION}.aliyuncs.com"
SAMPLE_RATE = settings.TINGWU_SAMPLE_RATE or 16000


def _acs() -> AcsClient:
    ak = settings.ALIBABA_CLOUD_ACCESS_KEY_ID
    sk = settings.ALIBABA_CLOUD_ACCESS_KEY_SECRET
    if not (ak and sk):
        raise EnvironmentError(
            "缺少 AK/SK：ALIBABA_CLOUD_ACCESS_KEY_ID / SECRET"
        )
    return AcsClient(ak, sk, REGION)


def _create_task():
    appkey = settings.ALIBABA_TINGWU_APPKEY
    if not appkey:
        raise EnvironmentError("缺少 ALIBABA_TINGWU_APPKEY")
    client = _acs()
    req = CommonRequest()
    req.set_domain(DOMAIN)
    req.set_version("2023-09-30")
    req.set_protocol_type("https")
    req.set_method("PUT")
    req.set_uri_pattern("/openapi/tingwu/v2/tasks")
    req.add_query_param("type", "realtime")
    req.add_header("Content-Type", "application/json")
    body = {
        "AppKey": appkey,
        "Input": {
            "Format": settings.TINGWU_FORMAT or "pcm",
            "SampleRate": SAMPLE_RATE,
            "SourceLanguage": settings.TINGWU_LANG or "cn",
        },
        "Parameters": {"Transcription": {"OutputLevel": 2}},
    }
    req.set_content(json.dumps(body).encode("utf-8"))
    resp = json.loads(client.do_action_with_exception(req))
    data = resp["Data"]
    return data["MeetingJoinUrl"], data["TaskId"]


def _stop_task(task_id: str):
    client = _acs()
    req = CommonRequest()
    req.set_domain(DOMAIN)
    req.set_version("2023-09-30")
    req.set_protocol_type("https")
    req.set_method("PUT")
    req.set_uri_pattern("/openapi/tingwu/v2/tasks")
    req.add_query_param("type", "realtime")
    req.add_query_param("operation", "stop")
    req.add_header("Content-Type", "application/json")
    body = {"AppKey": settings.ALIBABA_TINGWU_APPKEY, "Input": {"TaskId": task_id}}
    req.set_content(json.dumps(body).encode("utf-8"))
    return json.loads(client.do_action_with_exception(req))


class TingwuRealtimeClient:
    def __init__(self, audio_file: str):
        self.audio_file = audio_file
        self.result_text = ""
        self.ws_url, self.task_id = _create_task()
        self.__thread = threading.Thread(target=self._run)

    # --- 回调 ---
    def on_start(self, message, *args):
        print("[tingwu] streaming session started", flush=True)

    def on_sentence_begin(self, message, *args):
        payload = {}
        try:
            payload = json.loads(message).get("payload", {})
        except Exception:
            return
        text = payload.get("result") or payload.get("text")
        if text:
            print(f"[tingwu] ⇢ sentence begin: {text}", flush=True)

    def on_result_changed(self, message, *args):
        try:
            payload = json.loads(message).get("payload", {})
        except Exception:
            return
        text = payload.get("result") or payload.get("text")
        if text:
            print(f"[tingwu] … partial: {text}", flush=True)

    def on_sentence_end(self, message, *args):
        try:
            payload = json.loads(message).get("payload", {})
            text = payload.get("result", "")
            if text:
                self.result_text += text + " "
                print(f"[tingwu] ✓ final: {text}", flush=True)
        except Exception:
            pass

    def on_completed(self, message, *args):
        print("[tingwu] streaming session completed", flush=True)

    def on_error(self, message, *args):
        print(f"[tingwu] ! error: {message}", file=sys.stderr, flush=True)

    def on_close(self, *args):
        print("[tingwu] connection closed", flush=True)

    def _run(self):
        rm = nls.NlsRealtimeMeeting(
            url=self.ws_url,
            on_start=self.on_start,
            on_sentence_begin=self.on_sentence_begin,
            on_result_changed=self.on_result_changed,
            on_sentence_end=self.on_sentence_end,
            on_completed=self.on_completed,
            on_error=self.on_error,
            on_close=self.on_close,
            callback_args=[self.task_id],
        )
        rm.start()
        # 推流音频：640字节一包（10ms @16k/mono/s16le）；若源是 wav，建议先转 PCM。
        with open(self.audio_file, "rb") as f:
            data = f.read()
        total = max((len(data) + 639) // 640, 1)
        for idx in range(0, len(data), 640):
            chunk_no = idx // 640 + 1
            rm.send_audio(data[idx : idx + 640])
            print(f"[tingwu] ↳ streamed chunk {chunk_no}/{total}", flush=True)
            time.sleep(0.01)
        rm.stop()
        time.sleep(1.5)

    def transcribe(self) -> str:
        self.__thread.start()
        self.__thread.join()
        try:
            _stop_task(self.task_id)
        except Exception:
            pass
        return self.result_text.strip()


def transcribe(file_path: str) -> str:
    """外部直接调用的便捷函数"""

    return TingwuRealtimeClient(file_path).transcribe()

