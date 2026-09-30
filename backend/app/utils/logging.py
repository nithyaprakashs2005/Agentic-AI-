"""Structured JSON logging with correlation IDs.

system specification Standard #9: every log entry carries conversation_id, customer_id,
channel, and langgraph_run_id where known. No `print()` anywhere in the app.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import MutableMapping
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# Correlation context, populated per-request / per-turn.
_conversation_id: ContextVar[str | None] = ContextVar("conversation_id", default=None)
_customer_id: ContextVar[str | None] = ContextVar("customer_id", default=None)
_channel: ContextVar[str | None] = ContextVar("channel", default=None)
_run_id: ContextVar[str | None] = ContextVar("langgraph_run_id", default=None)

# Every attribute a LogRecord already owns. Derived, not hand-listed, so it
# cannot drift across Python versions.
_RESERVED_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime"}


def bind_context(
    *,
    conversation_id: str | None = None,
    customer_id: str | None = None,
    channel: str | None = None,
    langgraph_run_id: str | None = None,
) -> None:
    """Bind correlation IDs for the current async context."""
    if conversation_id is not None:
        _conversation_id.set(conversation_id)
    if customer_id is not None:
        _customer_id.set(customer_id)
    if channel is not None:
        _channel.set(channel)
    if langgraph_run_id is not None:
        _run_id.set(langgraph_run_id)


def current_context() -> dict[str, str]:
    ctx = {
        "conversation_id": _conversation_id.get(),
        "customer_id": _customer_id.get(),
        "channel": _channel.get(),
        "langgraph_run_id": _run_id.get(),
    }
    return {k: v for k, v in ctx.items() if v is not None}


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:16]}"


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line, merging correlation context and extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(current_context())

        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Uvicorn's own handlers would double-print; route them through root.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    # LiteLLM is extremely chatty at INFO.
    logging.getLogger("LiteLLM").setLevel("WARNING")
    logging.getLogger("httpx").setLevel("WARNING")


class SafeExtraLogger(logging.LoggerAdapter[logging.Logger]):
    """Logger that cannot be crashed by a colliding `extra` key.

    Python's logging raises `KeyError("Attempt to overwrite 'name' in
    LogRecord")` when `extra` carries a reserved attribute name -- `name`,
    `module`, `filename`, `args`, `levelname`, and friends. That turns a log
    line into an exception, and if the call sits inside a `try`, the failure
    masquerades as whatever that block was doing.

    This bit: `log.info("qdrant.collection_created", extra={"name": ...})` threw
    inside `QdrantHybridStore.connect`, was caught by its `except Exception`,
    and reported a perfectly healthy Qdrant as unavailable.

    Colliding keys are renamed with a `x_` prefix rather than dropped, so the
    value still reaches the log instead of disappearing silently.
    """

    def process(
        self, msg: object, kwargs: MutableMapping[str, Any]
    ) -> tuple[object, MutableMapping[str, Any]]:
        extra = kwargs.get("extra")
        if extra:
            kwargs["extra"] = {
                (f"x_{k}" if k in _RESERVED_ATTRS else k): v for k, v in extra.items()
            }
        return msg, kwargs


def get_logger(name: str) -> SafeExtraLogger:
    return SafeExtraLogger(logging.getLogger(name), {})
