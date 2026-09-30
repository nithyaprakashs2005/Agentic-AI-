"""Span tracing across graph nodes, MCP tools, and model calls.

system specification Phase 7 asks for Langfuse span tracing over every LangGraph node and
MCP tool. Langfuse is an optional extra and is not installed here, so the design
separates *instrumentation* from *export*:

* The instrumentation is real and always runs -- nodes, tools, and model calls
  open spans regardless of backend.
* The backend is swappable. `LangfuseTracer` exports when configured;
  `RecordingTracer` keeps spans in memory (used by tests and `/api/traces`, so
  the instrumentation is verifiable without Langfuse); `NoopTracer` costs
  nothing when tracing is off.

That split means the tracing is testable now and needs only a backend swap to
reach Langfuse -- rather than being untestable code that has never run.

Spans carry no message content by default. A trace is a debugging artefact that
often outlives the conversation, and system specification forbids persisting raw PII, so
only metadata (names, timings, counts, providers) is recorded unless
`TRACING_CAPTURE_CONTENT` is explicitly enabled.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from app.config import Settings, get_settings
from app.utils.logging import get_logger

log = get_logger(__name__)

SpanKind = Literal["node", "tool", "generation", "retrieval", "request"]

MAX_SPANS = 2_000

# Current parent span id, so nested spans form a tree without threading an
# explicit context object through every call site.
_current_span: ContextVar[str | None] = ContextVar("current_span", default=None)
_current_trace: ContextVar[str | None] = ContextVar("current_trace", default=None)


@dataclass(slots=True)
class Span:
    id: str
    trace_id: str
    parent_id: str | None
    name: str
    kind: SpanKind
    started_at: float
    ended_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # perf_counter is monotonic and right for durations, but a backend displays
    # wall-clock times, so both are recorded.
    started_wall: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_wall: datetime | None = None

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.perf_counter()
        return round((end - self.started_at) * 1000, 3)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "error": self.error,
        }


class Tracer(Protocol):
    backend: str

    def emit(self, span: Span) -> None: ...
    def describe(self) -> dict[str, Any]: ...


class NoopTracer:
    """Discards spans. Used when tracing is disabled."""

    backend = "disabled"

    def emit(self, span: Span) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend, "exporting": False}


class RecordingTracer:
    """Keeps recent spans in memory.

    The default when tracing is enabled but Langfuse is unavailable -- which
    makes the instrumentation inspectable via `/api/traces` instead of silently
    doing nothing.
    """

    backend = "in-memory"

    def __init__(self, max_spans: int = MAX_SPANS) -> None:
        self._spans: deque[Span] = deque(maxlen=max_spans)

    def emit(self, span: Span) -> None:
        self._spans.append(span)

    def spans(self, trace_id: str | None = None, limit: int = 200) -> list[Span]:
        rows = [s for s in self._spans if trace_id is None or s.trace_id == trace_id]
        return rows[-limit:]

    def traces(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent traces, newest first, with their root span and total duration."""
        grouped: dict[str, list[Span]] = {}
        for span in self._spans:
            grouped.setdefault(span.trace_id, []).append(span)

        summaries: list[dict[str, Any]] = [
            {
                "trace_id": trace_id,
                "spans": len(spans),
                "root": next((s.name for s in spans if s.parent_id is None), spans[0].name),
                "duration_ms": round(max(s.duration_ms for s in spans) if spans else 0.0, 3),
                "errors": sum(1 for s in spans if s.error),
                "started_at": min(s.started_at for s in spans),
            }
            for trace_id, spans in grouped.items()
        ]
        return sorted(summaries, key=lambda s: -float(s["started_at"]))[:limit]

    def clear(self) -> None:
        self._spans.clear()

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "exporting": False,
            "spans_buffered": len(self._spans),
            "capacity": self._spans.maxlen,
            "note": "spans are inspectable at /api/traces but are not exported",
        }


class LangfuseTracer:
    """Exports spans to a Langfuse instance."""

    backend = "langfuse"

    def __init__(self, client: Any, host: str) -> None:
        self._client = client
        self._host = host

    @classmethod
    def connect(cls, settings: Settings) -> LangfuseTracer | None:
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            return None
        try:
            from langfuse import Langfuse
        except ImportError:
            log.info(
                "tracing.langfuse_not_installed", extra={"hint": "uv sync --extra observability"}
            )
            return None

        try:
            client = Langfuse(
                public_key=settings.langfuse_public_key.get_secret_value(),
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                host=settings.langfuse_host,
            )
            log.info("tracing.langfuse_connected", extra={"host": settings.langfuse_host})
            return cls(client, settings.langfuse_host)
        except Exception as exc:
            log.warning("tracing.langfuse_unavailable", extra={"error": str(exc)[:200]})
            return None

    def emit(self, span: Span) -> None:
        try:
            if span.parent_id is None:
                # The trace has to exist before observations can hang off it.
                self._client.trace(id=span.trace_id, name=span.name)

            self._client.span(
                id=span.id,
                trace_id=span.trace_id,
                parent_observation_id=span.parent_id,
                name=span.name,
                start_time=span.started_wall,
                end_time=span.ended_wall,
                metadata={**span.metadata, "kind": span.kind},
                level="ERROR" if span.error else "DEFAULT",
                status_message=span.error,
            )
        except Exception:
            log.warning("tracing.emit_failed", exc_info=True)

    def flush(self) -> None:
        """Send anything still buffered. Called on shutdown."""
        try:
            self._client.flush()
        except Exception:
            log.warning("tracing.flush_failed", exc_info=True)

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend, "exporting": True, "host": self._host}


_tracer: Tracer = NoopTracer()


def init_tracing(settings: Settings | None = None) -> Tracer:
    """Choose a backend: Langfuse when configured, else in-memory, else off."""
    global _tracer
    settings = settings or get_settings()

    if not settings.tracing_enabled:
        _tracer = NoopTracer()
    else:
        _tracer = LangfuseTracer.connect(settings) or RecordingTracer()

    log.info("tracing.ready", extra=_tracer.describe())
    return _tracer


def get_tracer() -> Tracer:
    return _tracer


async def shutdown_tracing() -> None:
    """Flush buffered spans so a clean shutdown does not lose the tail."""
    flush = getattr(_tracer, "flush", None)
    if callable(flush):
        flush()
        log.info("tracing.flushed")


def new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:16]}"


@contextmanager
def start_trace(trace_id: str | None = None) -> Iterator[str]:
    """Open a root trace for one turn."""
    resolved = trace_id or new_trace_id()
    trace_token = _current_trace.set(resolved)
    span_token = _current_span.set(None)
    try:
        yield resolved
    finally:
        _current_span.reset(span_token)
        _current_trace.reset(trace_token)


@contextmanager
def span(name: str, kind: SpanKind = "node", **metadata: Any) -> Iterator[Span]:
    """Record a span around a block.

    Safe to use when tracing is disabled: the span object is still yielded so
    callers can attach metadata unconditionally, it is simply never emitted.
    """
    tracer = get_tracer()
    record = Span(
        id=f"span_{uuid.uuid4().hex[:12]}",
        trace_id=_current_trace.get() or new_trace_id(),
        parent_id=_current_span.get(),
        name=name,
        kind=kind,
        started_at=time.perf_counter(),
        metadata=dict(metadata),
    )
    token = _current_span.set(record.id)
    try:
        yield record
    except Exception as exc:
        record.error = f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        _current_span.reset(token)
        record.ended_at = time.perf_counter()
        record.ended_wall = datetime.now(UTC)
        tracer.emit(record)


def capture_content(settings: Settings | None = None) -> bool:
    """Whether spans may carry message text.

    Off by default: traces outlive conversations and system specification forbids
    persisting raw PII.
    """
    settings = settings or get_settings()
    return bool(settings.tracing_capture_content)
