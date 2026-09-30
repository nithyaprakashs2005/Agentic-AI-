"""Router node classification and cross-channel session continuity."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from app.agents.nodes.router import router_node
from app.agents.state import initial_state
from app.services.session import (
    InMemorySessions,
    Session,
    init_sessions,
    resolve_session,
)


def state_with(message: str) -> Any:
    state = initial_state(customer_id="cust_demo", channel="webchat")
    state["messages"] = [HumanMessage(content=message)]
    return state


# --- Router -------------------------------------------------------------
@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("I was charged twice for my subscription", "billing"),
        ("the app crashes when I try to log in", "technical"),
        ("I need to return a damaged item", "returns"),
        ("where is my package", "orders"),
    ],
)
async def test_router_classifies_intent(message: str, intent: str) -> None:
    update = await router_node(state_with(message))
    assert update["intent"] == intent


async def test_router_owns_resolution_path_and_model_tier() -> None:
    simple = await router_node(state_with("how long does a refund take"))
    assert simple["resolution_path"] == "simple"
    assert simple["model_tier"] == "slm"

    complex_ = await router_node(state_with("I want to dispute an unauthorized charge"))
    assert complex_["resolution_path"] == "complex"
    assert complex_["model_tier"] == "frontier"


async def test_router_resets_the_per_turn_tool_budget() -> None:
    update = await router_node(state_with("where is my order"))
    assert update["tool_attempts"] == 0


async def test_router_handles_an_empty_turn() -> None:
    state = initial_state(customer_id="cust_demo", channel="webchat")
    update = await router_node(state)
    assert update["intent"] == "general"
    assert update["intent_confidence"] == 0.0


async def test_router_records_timing_and_provider() -> None:
    update = await router_node(state_with("how long does a refund take"))
    assert "router" in update["node_timings_ms"]
    assert update["llm_stubbed"] is True


async def test_router_never_sends_raw_pii_to_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """system specification Constraints: PII must be masked before inference."""
    seen: list[str] = []

    import app.agents.nodes.router as router_module

    original = router_module.get_gateway()

    class Spy:
        async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
            seen.extend(m["content"] for m in messages)
            return await original.complete(messages, **kwargs)

    monkeypatch.setattr(router_module, "get_gateway", lambda: Spy())
    await router_node(state_with("charge on card 4111 1111 1111 1111 is wrong"))

    joined = " ".join(seen)
    assert "4111 1111 1111 1111" not in joined
    assert "[[CC_1]]" in joined


async def test_router_rejects_an_out_of_vocabulary_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hallucinated category becomes low-confidence general, not a bad route."""
    import app.agents.nodes.router as router_module
    from app.services.llm import LLMResult

    class Bogus:
        async def complete(self, *_a: Any, **_k: Any) -> LLMResult:
            return LLMResult(
                text='{"intent": "quantum_billing", "confidence": 0.99, '
                '"resolution_path": "simple"}',
                provider="fake",
                model="fake",
                latency_ms=1.0,
            )

    monkeypatch.setattr(router_module, "get_gateway", lambda: Bogus())
    update = await router_node(state_with("something odd"))
    assert update["intent"] == "general"
    assert update["intent_confidence"] == 0.0


@pytest.mark.parametrize("raw_confidence", ["not a number", None, 5.0, -2.0])
async def test_router_clamps_a_malformed_confidence(
    raw_confidence: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import app.agents.nodes.router as router_module
    from app.services.llm import LLMResult

    class Weird:
        async def complete(self, *_a: Any, **_k: Any) -> LLMResult:
            return LLMResult(
                text=json.dumps(
                    {"intent": "billing", "confidence": raw_confidence, "resolution_path": "simple"}
                ),
                provider="fake",
                model="fake",
                latency_ms=1.0,
            )

    monkeypatch.setattr(router_module, "get_gateway", lambda: Weird())
    update = await router_node(state_with("my bill is wrong"))
    assert 0.0 <= update["intent_confidence"] <= 1.0


# --- Sessions -----------------------------------------------------------
async def test_explicit_conversation_id_wins() -> None:
    await init_sessions()
    session = await resolve_session(
        customer_id="cust_demo", channel="webchat", conversation_id="explicit-1"
    )
    assert session.conversation_id == "explicit-1"


async def test_a_returning_customer_rejoins_their_thread() -> None:
    """The chat-then-call handover requirement."""
    await init_sessions()
    first = await resolve_session(customer_id="cust_demo", channel="webchat")
    second = await resolve_session(customer_id="cust_demo", channel="voice")

    assert second.conversation_id == first.conversation_id
    assert second.channels == ["webchat", "voice"]
    assert second.turn_count == 2


async def test_different_customers_get_different_threads() -> None:
    await init_sessions()
    a = await resolve_session(customer_id="cust_demo", channel="webchat")
    b = await resolve_session(customer_id="cust_abc123", channel="webchat")
    assert a.conversation_id != b.conversation_id


async def test_anonymous_turns_never_share_a_thread() -> None:
    await init_sessions()
    a = await resolve_session(customer_id="", channel="webchat")
    b = await resolve_session(customer_id="", channel="webchat")
    assert a.conversation_id != b.conversation_id


async def test_expired_sessions_are_swept() -> None:
    store = InMemorySessions(ttl=0)
    await store.put(Session(conversation_id="old", customer_id="cust_demo"))
    assert await store.get("old") is None
    assert await store.find_by_customer("cust_demo") is None


async def test_channel_list_does_not_duplicate() -> None:
    session = Session(conversation_id="c", customer_id="cust_demo")
    session.touch("webchat")
    session.touch("webchat")
    assert session.channels == ["webchat"]
    assert session.turn_count == 2


async def test_describe_reports_the_backend() -> None:
    backend = await init_sessions()
    assert backend.describe()["backend"] == "in-memory"
