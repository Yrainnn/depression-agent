from __future__ import annotations

from pathlib import Path
from typing import List

import sys
import types

import pytest

httpx_stub = types.ModuleType("httpx")


class _HTTPStatusError(Exception):
    pass


class _RequestError(Exception):
    pass


httpx_stub.Client = object
httpx_stub.HTTPStatusError = _HTTPStatusError
httpx_stub.RequestError = _RequestError
sys.modules.setdefault("httpx", httpx_stub)

config_stub = types.ModuleType("packages.common.config")


class _Settings:
    deepseek_api_base: str | None = None
    deepseek_api_key: str | None = None


config_stub.settings = _Settings()

packages_stub = types.ModuleType("packages")
common_stub = types.ModuleType("packages.common")
packages_stub.common = common_stub
common_stub.config = config_stub

sys.modules.setdefault("packages", packages_stub)
sys.modules.setdefault("packages.common", common_stub)
sys.modules.setdefault("packages.common.config", config_stub)

pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)

    @classmethod
    def model_validate(cls, value):
        return cls(**value)


def Field(default=None, **_kwargs):  # pragma: no cover - simple stub
    return default


pydantic_stub.BaseModel = _BaseModel
pydantic_stub.Field = Field

sys.modules.setdefault("pydantic", pydantic_stub)

tenacity_stub = types.ModuleType("tenacity")


def retry(*_args, **_kwargs):  # pragma: no cover - simple stub
    def decorator(func):
        return func

    return decorator


def stop_after_attempt(*_args, **_kwargs):  # pragma: no cover - simple stub
    return None


def wait_fixed(*_args, **_kwargs):  # pragma: no cover - simple stub
    return None


tenacity_stub.retry = retry
tenacity_stub.stop_after_attempt = stop_after_attempt
tenacity_stub.wait_fixed = wait_fixed

sys.modules.setdefault("tenacity", tenacity_stub)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.llm.json_client import DeepSeekJSONClient


class _DummyResponse:
    def raise_for_status(self) -> None:  # pragma: no cover - simple stub
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "{}"}}]}


class _DummyClient:
    def __init__(self, *_, captured_urls: List[str] | None = None, **__):
        self._captured = captured_urls if captured_urls is not None else []

    def __enter__(self) -> "_DummyClient":
        return self

    def __exit__(self, *exc_info) -> None:  # pragma: no cover - simple stub
        return None

    def post(self, url: str, **_) -> _DummyResponse:
        self._captured.append(url)
        return _DummyResponse()


@pytest.mark.parametrize(
    "base",
    [
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    ],
)
def test_post_chat_normalizes_deepseek_base(monkeypatch: pytest.MonkeyPatch, base: str) -> None:
    captured: List[str] = []

    def _client_factory(*args, **kwargs):
        kwargs.setdefault("captured_urls", captured)
        return _DummyClient(*args, **kwargs)

    monkeypatch.setattr("services.llm.json_client.httpx.Client", _client_factory)

    client = DeepSeekJSONClient(base=base, key="test-key", model="dummy")

    result = client._post_chat(messages=[{"role": "user", "content": "hi"}])

    assert result == "{}"
    assert captured == ["https://api.deepseek.com/v1/chat/completions"]
