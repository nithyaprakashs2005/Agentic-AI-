"""Trace inspection endpoints.

Exists so the instrumentation is verifiable without a Langfuse instance: when
tracing falls back to the in-memory recorder, spans are still inspectable here.
With Langfuse exporting, these return an empty view and point at the Langfuse UI.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.services.tracing import RecordingTracer, get_tracer

router = APIRouter(prefix="/api/traces", tags=["traces"])


@router.get("")
async def list_traces(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    """Recent traces, newest first."""
    tracer = get_tracer()
    if not isinstance(tracer, RecordingTracer):
        return {
            "traces": [],
            "backend": tracer.backend,
            "note": _backend_note(tracer.backend),
        }
    return {"traces": tracer.traces(limit=limit), "backend": tracer.backend}


@router.get("/status")
async def tracing_status() -> dict[str, Any]:
    """What the tracing backend is, and what it would take to export."""
    settings = get_settings()
    tracer = get_tracer()
    missing: list[str] = []
    if not settings.langfuse_public_key:
        missing.append("LANGFUSE_PUBLIC_KEY")
    if not settings.langfuse_secret_key:
        missing.append("LANGFUSE_SECRET_KEY")

    return {
        **tracer.describe(),
        "enabled": settings.tracing_enabled,
        "captures_message_content": settings.tracing_capture_content,
        "missing_configuration": missing,
        "instrumented": ["graph nodes", "MCP tools", "model calls"],
    }


@router.get("/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """Every span in one trace, in the order recorded."""
    tracer = get_tracer()
    if not isinstance(tracer, RecordingTracer):
        raise HTTPException(
            status_code=404,
            detail=_backend_note(tracer.backend),
        )

    spans = tracer.spans(trace_id=trace_id)
    if not spans:
        raise HTTPException(status_code=404, detail="Unknown trace")

    return {
        "trace_id": trace_id,
        "span_count": len(spans),
        "total_ms": round(max(s.duration_ms for s in spans), 3),
        "spans": [s.public() for s in spans],
    }


def _backend_note(backend: str) -> str:
    if backend == "langfuse":
        return "spans are exported to Langfuse; inspect them in the Langfuse UI"
    return "tracing is disabled; set TRACING_ENABLED=true to record spans"
