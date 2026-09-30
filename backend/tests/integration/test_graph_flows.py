"""End-to-end orchestrator flows through the compiled graph."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.agents.state import initial_state

pytestmark = pytest.mark.timeout(60)


async def run(
    graph: Any,
    message: str,
    *,
    thread: str,
    customer_id: str = "cust_demo",
    tier: str = "standard",
    channel: str = "webchat",
) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread}}
    snapshot = await graph.aget_state(config)
    if snapshot.values:
        payload: dict[str, Any] = {"messages": [HumanMessage(content=message)]}
    else:
        seeded = initial_state(
            customer_id=customer_id,
            channel=channel,  # type: ignore[arg-type]
            conversation_id=thread,
            customer_tier=tier,  # type: ignore[arg-type]
        )
        seeded["messages"] = [HumanMessage(content=message)]
        payload = dict(seeded)
    return await graph.ainvoke(payload, config=config)


# --- Happy path ---------------------------------------------------------
async def test_faq_resolves_without_escalation(compiled_graph: Any) -> None:
    state = await run(compiled_graph, "How long do refunds take to appear?", thread="t1")
    assert state["intent"] == "billing"
    assert state["resolution_path"] == "simple"
    assert state["escalation_required"] is False
    assert state["resolution_status"] == "resolved"
    assert state["ticket_id"]


async def test_every_answer_carries_citations(compiled_graph: Any) -> None:
    """system specification: RAG-grounded responses must include verifiable sources."""
    state = await run(compiled_graph, "What is the return window?", thread="t2")
    assert state["rag_citations"]
    assert all(c["source"].startswith("http") for c in state["rag_citations"])


async def test_intent_drives_the_retrieved_category(compiled_graph: Any) -> None:
    state = await run(compiled_graph, "My authenticator codes are rejected", thread="t3")
    assert state["intent"] == "technical"
    assert any(c["category"] == "technical" for c in state["rag_citations"])


async def test_greeting_neither_escalates_nor_retrieves(compiled_graph: Any) -> None:
    state = await run(compiled_graph, "hello", thread="t4")
    assert state["escalation_required"] is False
    assert state["rag_citations"] == []


async def test_order_agent_returns_verified_tracking_details(compiled_graph: Any) -> None:
    state = await run(compiled_graph, "Where is my order #10245?", thread="t-order")
    assert state["intent"] == "orders"
    assert state["resolution_status"] == "resolved"
    assert "ORD-10245" in state["draft_response"]
    assert "Estimated delivery: 2026-09-26" in state["draft_response"]


async def test_return_is_policy_checked_before_confirmation(compiled_graph: Any) -> None:
    state = await run(
        compiled_graph, "I want to return the shoes I bought last week.", thread="t-return"
    )
    assert state["intent"] == "returns"
    assert state["policy_confirmed"] is True
    assert state["return_eligibility"]["eligible"] is True
    assert "return is confirmed" in state["draft_response"].lower()


# --- Escalation paths ---------------------------------------------------
async def test_explicit_human_request_escalates_and_parks(compiled_graph: Any) -> None:
    state = await run(compiled_graph, "I want to speak to a human", thread="t5")
    assert state["escalation_required"] is True
    assert state["escalation_reason"] == "explicit_keyword:human"
    assert state["resolution_status"] == "escalated"
    # The HITL gate parked the run.
    assert "__interrupt__" in state


async def test_handoff_packet_is_populated(compiled_graph: Any) -> None:
    state = await run(
        compiled_graph,
        "I am furious about this duplicate charge and want to dispute it",
        thread="t6",
        customer_id="cust_abc123",
        tier="enterprise",
    )
    packet = state["handoff_packet"]
    for key in (
        "executive_summary",
        "issue",
        "history",
        "actions_taken",
        "actions_attempted",
        "rag_citations",
        "sentiment_trajectory",
        "churn_risk",
        "recommended_actions",
        "resolution_path_used",
        "transcript",
    ):
        assert key in packet, f"handoff packet missing {key}"
    assert packet["customer_tier"] == "enterprise"
    assert packet["resolution_path_used"] in ("simple", "complex")
    assert packet["recommended_actions"]


async def test_complex_path_produces_a_plan(compiled_graph: Any) -> None:
    state = await run(
        compiled_graph,
        "I need to dispute an unauthorized charge, this is the third time",
        thread="t7",
        customer_id="cust_abc123",
        tier="enterprise",
    )
    assert state["resolution_path"] == "complex"
    assert state["deep_agent_todos"]
    assert state["tool_attempts"] > 0


async def test_ticket_is_created_before_the_run_parks(compiled_graph: Any) -> None:
    """The human must find a CRM record waiting, not create one themselves."""
    state = await run(compiled_graph, "get me a manager", thread="t8")
    assert state["ticket_id"]

    from app.services.crm import get_crm

    ticket = await get_crm().get_ticket(state["ticket_id"])
    assert ticket is not None
    assert ticket.status == "pending_human"
    assert ticket.handoff_packet is not None


async def test_resume_after_interrupt_resolves(compiled_graph: Any) -> None:
    config = {"configurable": {"thread_id": "t9"}}
    await run(compiled_graph, "I want a human", thread="t9")

    snapshot = await compiled_graph.aget_state(config)
    assert snapshot.next, "expected the graph to be parked awaiting a human"

    resumed = await compiled_graph.ainvoke(
        Command(resume="Resolved by agent Dana: goodwill credit applied"), config=config
    )
    assert resumed["resolution_status"] == "resolved"
    assert resumed["escalation_required"] is False


# --- Authentication gate ------------------------------------------------
async def test_unknown_customer_gets_a_challenge_not_an_answer(compiled_graph: Any) -> None:
    state = await run(compiled_graph, "where is my order", thread="t10", customer_id="ghost")
    assert state["authenticated"] is False
    assert state["auth_attempts"] == 1
    assert "verify" in state["draft_response"].lower()
    # The gate must stop the turn before resolution.
    assert state["rag_citations"] == []
    assert state["ticket_id"] is None


async def test_challenge_response_authenticates(compiled_graph: Any) -> None:
    await run(compiled_graph, "where is my order", thread="t11", customer_id="ghost")
    state = await run(compiled_graph, "my email is demo@example.com", thread="t11")
    assert state["authenticated"] is True
    assert state["customer_id"] == "cust_demo"


async def test_repeated_auth_failure_escalates(compiled_graph: Any) -> None:
    for _ in range(3):
        state = await run(compiled_graph, "no idea sorry", thread="t12", customer_id="ghost")
    assert state["escalation_required"] is True
    assert state["escalation_reason"] == "identity_verification_failed"


async def test_enterprise_tier_is_read_from_the_crm(compiled_graph: Any) -> None:
    """Tier drives the escalation rule, so it must come from the CRM not the client."""
    state = await run(
        compiled_graph,
        "how long do refunds take",
        thread="t13",
        customer_id="cust_abc123",
        tier="free",  # client claims free; CRM says enterprise
    )
    assert state["customer_tier"] == "enterprise"


# --- Conversation continuity -------------------------------------------
async def test_state_persists_across_turns(compiled_graph: Any) -> None:
    await run(compiled_graph, "How long do refunds take?", thread="t14")
    second = await run(compiled_graph, "And what about bank transfers?", thread="t14")
    # Both turns are on one thread, so the transcript accumulates.
    assert len(second["messages"]) >= 4
    assert len(second["sentiment_trajectory"]) == 2


async def test_tool_attempts_do_not_accumulate_across_turns(compiled_graph: Any) -> None:
    """A long healthy chat must not trip `tool_attempts_exhausted`."""
    for _ in range(5):
        state = await run(compiled_graph, "Where is my order?", thread="t15")
    assert state["escalation_reason"] != "tool_attempts_exhausted"
    assert state["tool_attempts"] <= 2


async def test_voice_channel_reply_is_speakable(compiled_graph: Any) -> None:
    state = await run(compiled_graph, "How long do refunds take?", thread="t16", channel="voice")
    from app.utils.channel_adapter import to_voice

    spoken = to_voice(state["draft_response"])
    for forbidden in ("**", "##", "http", "$", "@"):
        assert forbidden not in spoken


async def test_node_timings_are_recorded_for_every_node(compiled_graph: Any) -> None:
    state = await run(compiled_graph, "How long do refunds take?", thread="t17")
    timings = state["node_timings_ms"]
    # The reducer must merge, not clobber -- several nodes ran this turn.
    assert len(timings) >= 4
    assert all(isinstance(v, float) for v in timings.values())
