from __future__ import annotations

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest

from apps.api.router_dm import router as dm_router
from services.store.repository import repository
from apps.api.router_asr_tingwu import router as tingwu_asr_router


app = FastAPI(title="Depression Agent API", version="0.1.0")
app.include_router(dm_router)
app.include_router(tingwu_asr_router) 

_registry = CollectorRegistry()
_health_gauge = Gauge("depression_agent_health", "Health status of the API", registry=_registry)
_health_gauge.set(1)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/metrics")
def metrics() -> Response:
    data = generate_latest(_registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/debug/redis")
def debug_redis() -> JSONResponse:
    if repository.ping():
        return JSONResponse({"redis": "pong"})
    return JSONResponse({"error": "redis unreachable"}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
