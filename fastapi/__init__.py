"""Minimal FastAPI stub for offline testing environments."""

from typing import Any, Callable, Optional


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class Request:
    async def json(self) -> Any:
        raise RuntimeError("Request stub does not support body parsing")


class UploadFile:
    filename: Optional[str] = None

    async def read(self) -> bytes:
        raise RuntimeError("UploadFile stub does not implement read")


def File(*_args, **kwargs):
    return kwargs.get("default")


def Form(*_args, **kwargs):
    return kwargs.get("default")


class _RouteMixin:
    def get(self, *_args, **_kwargs) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def post(self, *_args, **_kwargs) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


class APIRouter(_RouteMixin):
    def __init__(self, *args, **kwargs) -> None:
        pass


class FastAPI(_RouteMixin):
    def __init__(self, *args, **kwargs) -> None:
        self.routers = []

    def include_router(self, router: APIRouter) -> None:
        self.routers.append(router)


class _Status:
    HTTP_500_INTERNAL_SERVER_ERROR = 500


status = _Status()


__all__ = [
    "APIRouter",
    "FastAPI",
    "File",
    "Form",
    "HTTPException",
    "Request",
    "UploadFile",
    "status",
]
