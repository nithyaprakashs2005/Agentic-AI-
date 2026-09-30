"""Span tracing: nesting, coverage of every node, PII posture, and endpoints."""

from __future__ import annotations

from typing import Any

import pytest

from app.config import Settings
from app.services.tracing import (
    LangfuseTracer,
    NoopTracer,
    RecordingTracer,
    Span,
    get_tracer,
    init_tracing,
    span,
    start_trace,
)
from tests.conftest import patch_settings

pytestmark = pytest.mark.timeout(90)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> RecordingTracer:
    tracer = RecordingTracer()
    monkeypatch.setattr("app.services.tracing._tracer", tracer)
    return tracer


# --- Backend selection --------------------------------------------------
def test_disabled_tracing_uses_the_noop_backend() -> None:
    tracer = init_tracing(Settings(_env_file=None, tracing_enabled=False))  # type: ignore[call-arg]
    assert isinstance(tracer, NoopTracer)
    assert tracer.describe()["exporting"] is False


def test_enabled_without_langfuse_falls_back_to_recording() -> None:
    """Instrumentation must remain inspectable rather than silently vanish."""
    tracer = init_tracing(Settings(_env_file=None, tracing_enabled=True))  # type: ignore[call-arg]
    assert isinstance(tracer, RecordingTracer)
    assert tracer.describe()["exporting"] is False


def test_langfuse_connect_returns_none_without_keys() -> None:
    assert LangfuseTracer.connect(Settings(_env_file=None)) is None  # type: ignore[call-arg]


def test_langfuse_connect_returns_none_when_the_sdk_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """langfuse is an optional extra; absence must degrade, not crash."""
    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langfuse":
            raise ImportError("simulated: extra not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, langfuse_public_key="pk", langfuse_secret_key="sk"
    )
    assert LangfuseTracer.connect(settings) is None


def test_langfuse_connect_returns_none_when_the_host_is_unreachable() -> None:
    """A dead observability backend must not take the application down."""
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        langfuse_host="http://127.0.0.1:1",
    )
    tracer = LangfuseTracer.connect(settings)
    # Either it refuses to construct, or it constructs and emits are swallowed;
    # both are acceptable, a raised exception is not.
    if tracer is not None:
        tracer.emit(
            Span(id="s", trace_id="t", parent_id=None, name="x", kind="node", started_at=0.0)
        )


# --- Span mechanics -----------------------------------------------------
def test_span_records_name_kind_and_duration(recorder: RecordingTracer) -> None:
    with start_trace(), span("work", kind="node", intent="billing"):
        pass

    recorded = recorder.spans()
    assert len(recorded) == 1
    assert recorded[0].name == "work"
    assert recorded[0].kind == "node"
    assert recorded[0].metadata["intent"] == "billing"
    assert recorded[0].duration_ms >= 0


def test_nested_spans_form_a_tree(recorder: RecordingTracer) -> None:
    with start_trace(), span("outer") as outer, span("inner"):
        pass

    by_name = {s.name: s for s in recorder.spans()}
    assert by_name["inner"].parent_id == outer.id
    assert by_name["outer"].parent_id is None
    assert by_name["inner"].trace_id == by_name["outer"].trace_id


def test_sibling_spans_share_a_parent(recorder: RecordingTracer) -> None:
    with start_trace(), span("root") as root:
        with span("a"):
            pass
        with span("b"):
            pass

    siblings = [s for s in recorder.spans() if s.name in {"a", "b"}]
    assert {s.parent_id for s in siblings} == {root.id}


def test_span_records_an_exception_and_reraises(recorder: RecordingTracer) -> None:
    with pytest.raises(ValueError, match="boom"), start_trace(), span("failing"):
        raise ValueError("boom")

    recorded = recorder.spans()[0]
    assert recorded.error is not None
    assert "ValueError" in recorded.error


def test_span_is_safe_when_tracing_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers attach metadata unconditionally; disabled tracing must not break."""
    monkeypatch.setattr("app.services.tracing._tracer", NoopTracer())
    with start_trace(), span("work") as s:
        s.metadata["k"] = "v"
    assert get_tracer().describe()["backend"] == "disabled"


def test_ring_buffer_bounds_memory() -> None:
    tracer = RecordingTracer(max_spans=3)
    for i in range(10):
        tracer.emit(
            Span(id=f"s{i}", trace_id="t", parent_id=None, name="x", kind="node", started_at=0.0)
        )
    assert len(tracer.spans()) == 3


def test_traces_summary_groups_by_trace(recorder: RecordingTracer) -> None:
    with start_trace("trace_a"), span("root_a"):
        with span("child"):
            pass
    with start_trace("trace_b"), span("root_b"):
        pass

    summaries = {t["trace_id"]: t for t in recorder.traces()}
    assert summaries["trace_a"]["spans"] == 2
    assert summaries["trace_a"]["root"] == "root_a"
    assert summaries["trace_b"]["spans"] == 1


# --- Coverage of the real graph -----------------------------------------
async def test_every_graph_node_emits_a_span(
    compiled_graph: Any, recorder: RecordingTracer
) -> None:
    """Nodes are wrapped centrally, so none can escape instrumentation."""
    from langchain_core.messages import HumanMessage

    from app.agents.state import initial_state

    state = initial_state(customer_id="cust_demo", channel="webchat", conversation_id="tr-1")
    state["messages"] = [HumanMessage(content="How long do refunds take?")]

    with start_trace():
        await compiled_graph.ainvoke(dict(state), config={"configurable": {"thread_id": "tr-1"}})

    names = {s.name for s in recorder.spans()}
    for node in ("router", "auth", "agentic_rag", "simple_resolution", "sentiment", "ticketing"):
        assert node in names, f"{node} emitted no span"


async def test_model_calls_emit_generation_spans(
    compiled_graph: Any, recorder: RecordingTracer
) -> None:
    from langchain_core.messages import HumanMessage

    from app.agents.state import initial_state

    state = initial_state(customer_id="cust_demo", channel="webchat", conversation_id="tr-2")
    state["messages"] = [HumanMessage(content="How long do refunds take?")]

    with start_trace():
        await compiled_graph.ainvoke(dict(state), config={"configurable": {"thread_id": "tr-2"}})

    generations = [s for s in recorder.spans() if s.kind == "generation"]
    assert generations, "expected generation spans for model calls"
    assert all(s.metadata.get("purpose") for s in generations)


async def test_mcp_tool_calls_emit_tool_spans(recorder: RecordingTracer, crm: Any) -> None:
    from app.mcp.tools import call_tool

    with start_trace():
        await call_tool("crm.order_status", {"order_id": "ORD-84021"})

    tool_spans = [s for s in recorder.spans() if s.kind == "tool"]
    assert [s.name for s in tool_spans] == ["crm.order_status"]
    assert tool_spans[0].metadata["server"] == "crm"


async def test_spans_do_not_carry_message_content(
    compiled_graph: Any, recorder: RecordingTracer
) -> None:
    """Traces outlive conversations; raw text must not be persisted into them."""
    from langchain_core.messages import HumanMessage

    from app.agents.state import initial_state

    secret = "my card is 4111 1111 1111 1111"
    state = initial_state(customer_id="cust_demo", channel="webchat", conversation_id="tr-3")
    state["messages"] = [HumanMessage(content=secret)]

    with start_trace():
        await compiled_graph.ainvoke(dict(state), config={"configurable": {"thread_id": "tr-3"}})

    dumped = str([s.public() for s in recorder.spans()])
    assert "4111" not in dumped
    assert secret not in dumped


# --- Endpoints ----------------------------------------------------------
async def test_status_reports_backend_and_what_is_missing(api_client: Any) -> None:
    body = (await api_client.get("/api/traces/status")).json()
    assert body["enabled"] is True
    assert body["exporting"] is False
    assert body["captures_message_content"] is False
    assert any("LANGFUSE_PUBLIC_KEY" in m for m in body["missing_configuration"])
    assert "graph nodes" in body["instrumented"]


async def test_a_chat_turn_produces_an_inspectable_trace(api_client: Any) -> None:
    await api_client.post(
        "/api/chat", json={"message": "How long do refunds take?", "conversation_id": "tr-api"}
    )

    listing = (await api_client.get("/api/traces")).json()
    assert listing["traces"], "expected at least one trace"

    trace_id = listing["traces"][0]["trace_id"]
    detail = (await api_client.get(f"/api/traces/{trace_id}")).json()
    assert detail["span_count"] > 1
    kinds = {s["kind"] for s in detail["spans"]}
    assert {"request", "node"} <= kinds


async def test_unknown_trace_is_404(api_client: Any) -> None:
    assert (await api_client.get("/api/traces/trace_nope")).status_code == 404


async def test_health_flags_langfuse_as_degraded(api_client: Any) -> None:
    body = (await api_client.get("/health")).json()
    assert any("Langfuse not configured" in d for d in body["degradations"])
    assert body["subsystems"]["tracing"]["backend"] == "in-memory"


def test_capture_content_defaults_off() -> None:
    from app.services.tracing import capture_content

    assert capture_content(Settings(_env_file=None)) is False  # type: ignore[call-arg]


def test_capture_content_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services.tracing import capture_content

    settings = Settings(_env_file=None, tracing_capture_content=True)  # type: ignore[call-arg]
    patch_settings(monkeypatch, settings)
    assert capture_content(settings) is True
