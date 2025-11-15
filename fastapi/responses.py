"""Minimal FastAPI response stubs for tests."""

from typing import Any, Optional


class Response:
    def __init__(self, content: Any = None, media_type: Optional[str] = None):
        self.content = content
        self.media_type = media_type


class JSONResponse(Response):
    def __init__(self, content: Any, status_code: int = 200):
        super().__init__(content=content, media_type="application/json")
        self.status_code = status_code


__all__ = ["JSONResponse", "Response"]
