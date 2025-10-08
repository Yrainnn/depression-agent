"""Asynchronous helpers for interacting with Tingwu realtime tasks."""

from __future__ import annotations

import asyncio
import json
from typing import Tuple

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

from packages.common.config import settings

REGION = settings.TINGWU_REGION or "cn-beijing"
DOMAIN = f"tingwu.{REGION}.aliyuncs.com"


def _acs() -> AcsClient:
    ak = settings.ALIBABA_CLOUD_ACCESS_KEY_ID
    sk = settings.ALIBABA_CLOUD_ACCESS_KEY_SECRET
    if not (ak and sk):
        raise EnvironmentError(
            "缺少 AK/SK：ALIBABA_CLOUD_ACCESS_KEY_ID / SECRET"
        )
    return AcsClient(ak, sk, REGION)


def _create_request() -> CommonRequest:
    req = CommonRequest()
    req.set_domain(DOMAIN)
    req.set_version("2023-09-30")
    req.set_protocol_type("https")
    req.set_method("PUT")
    req.set_uri_pattern("/openapi/tingwu/v2/tasks")
    req.add_query_param("type", "realtime")
    req.add_header("Content-Type", "application/json")
    body = {
        "AppKey": settings.ALIBABA_TINGWU_APPKEY,
        "Input": {
            "Format": settings.TINGWU_FORMAT or "pcm",
            "SampleRate": 16000,
            "SourceLanguage": settings.TINGWU_LANG or "cn",
        },
        "Parameters": {"Transcription": {"OutputLevel": 2}},
    }
    req.set_content(json.dumps(body).encode("utf-8"))
    return req


def _stop_request(task_id: str) -> CommonRequest:
    req = CommonRequest()
    req.set_domain(DOMAIN)
    req.set_version("2023-09-30")
    req.set_protocol_type("https")
    req.set_method("PUT")
    req.set_uri_pattern("/openapi/tingwu/v2/tasks")
    req.add_query_param("type", "realtime")
    req.add_query_param("operation", "stop")
    req.add_header("Content-Type", "application/json")
    body = {
        "AppKey": settings.ALIBABA_TINGWU_APPKEY,
        "Input": {"TaskId": task_id},
    }
    req.set_content(json.dumps(body).encode("utf-8"))
    return req


def _create_task_sync() -> Tuple[str, str]:
    client = _acs()
    response = json.loads(client.do_action_with_exception(_create_request()))
    print(
        "[tingwu] create_realtime_task response:",
        json.dumps(response, ensure_ascii=False),
        flush=True,
    )
    data = response.get("Data", {})
    meeting_url = data.get("MeetingJoinUrl", "")
    meeting_code = data.get("MeetingCode") or data.get("meeting_code")
    if meeting_url and "?mc=" not in meeting_url and meeting_code:
        separator = "&" if "?" in meeting_url else "?"
        meeting_url = f"{meeting_url}{separator}mc={meeting_code}"
    task_id = data.get("TaskId", "")
    if not meeting_url or not task_id:
        raise RuntimeError(
            "无法从听悟返回中获取 MeetingJoinUrl / TaskId"
        )
    return meeting_url, task_id


def _stop_task_sync(task_id: str) -> dict:
    client = _acs()
    response = json.loads(client.do_action_with_exception(_stop_request(task_id)))
    return response


async def create_realtime_task() -> Tuple[str, str]:
    """Create a Tingwu realtime task asynchronously."""

    return await asyncio.to_thread(_create_task_sync)


async def stop_realtime_task(task_id: str) -> dict:
    """Stop the Tingwu realtime task asynchronously."""

    return await asyncio.to_thread(_stop_task_sync, task_id)

