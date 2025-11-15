"""Shared pytest fixtures and external dependency stubs."""
from __future__ import annotations

import sys
import types
from pathlib import Path


def _register_module(name: str, module: types.ModuleType) -> None:
    """Register a stub module when the real dependency is absent."""

    sys.modules.setdefault(name, module)


# ---------------------------------------------------------------------------
# Alibaba TingWu / NLS SDK stubs so audio modules import without external deps.
_nls = types.ModuleType("nls")
_nls.enableTrace = lambda *args, **kwargs: None  # type: ignore[attr-defined]
_nls.setLogFile = lambda *args, **kwargs: None  # type: ignore[attr-defined]
_register_module("nls", _nls)


class _AcsClient:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - trivial stub
        pass

    def do_action_with_exception(self, *_args, **_kwargs) -> bytes:
        return b'{"Data": {"MeetingJoinUrl": "", "TaskId": ""}}'


_client_mod = types.ModuleType("aliyunsdkcore.client")
_client_mod.AcsClient = _AcsClient
_request_mod = types.ModuleType("aliyunsdkcore.request")


class _CommonRequest:
    def set_domain(self, *_args, **_kwargs) -> None:  # pragma: no cover - simple stub
        return None

    set_version = set_protocol_type = set_method = set_uri_pattern = set_domain
    add_query_param = add_header = set_content = set_domain


_request_mod.CommonRequest = _CommonRequest
_core_mod = types.ModuleType("aliyunsdkcore")
_core_mod.client = _client_mod
_core_mod.request = _request_mod

_register_module("aliyunsdkcore.client", _client_mod)
_register_module("aliyunsdkcore.request", _request_mod)
_register_module("aliyunsdkcore", _core_mod)


# ---------------------------------------------------------------------------
# Reporting pipeline stubs (jinja2 / weasyprint) for PDF generation imports.
_jinja2 = types.ModuleType("jinja2")


class _FakeTemplate:
    def render(self, **_: object) -> str:  # pragma: no cover - deterministic stub
        return "<html></html>"


class _FakeEnv:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def from_string(self, _template: str) -> _FakeTemplate:
        return _FakeTemplate()


_jinja2.Environment = lambda *args, **kwargs: _FakeEnv(*args, **kwargs)  # type: ignore[attr-defined]
_jinja2.select_autoescape = lambda *args, **kwargs: None  # type: ignore[attr-defined]
_register_module("jinja2", _jinja2)


_weasyprint = types.ModuleType("weasyprint")


class _FakeHTML:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def write_pdf(self, path: str) -> None:
        Path(path).write_bytes(b"PDF")


_weasyprint.HTML = lambda *args, **kwargs: _FakeHTML(*args, **kwargs)  # type: ignore[attr-defined]
_register_module("weasyprint", _weasyprint)

