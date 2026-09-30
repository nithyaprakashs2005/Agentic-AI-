"""Typed agent state for the main orchestrator.

Mirrors the `SupportAgentState` contract in system specification. Fields the escalation
rule engine reads (`customer_tier`, `intent_confidence`) are first-class here so
the rules cannot reference something the state does not carry.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

Channel = Literal["voice", "sms", "whatsapp", "webchat", "email"]
CustomerTier = Literal["free", "standard", "enterprise"]
ResolutionStatus = Literal["pending", "resolved", "escalated"]
ResolutionPath = Literal["simple", "complex"]
ModelTier = Literal["slm", "frontier"]

# The router's intent vocabulary. KB article categories must use these exact
# values, since retrieval filters `category` by `state["intent"]`.
INTENTS: tuple[str, ...] = ("billing", "technical", "returns", "orders", "general")


# Phrases that mean "stop talking to the robot". Checked verbatim, never by LLM,
# so an explicit request for a human can never be lost to a model judgement.
def merge_timings(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Reducer for per-node latency, so one node cannot clobber another's entry."""
    return {**(left or {}), **(right or {})}


ESCALATION_KEYWORDS: tuple[str, ...] = (
    "agent",
    "human",
    "representative",
    "real person",
    "someone else",
    "manager",
    "supervisor",
    "lawyer",
    "legal",
    "cancel my account",
    "close my account",
    "delete my account",
    "speak to a person",
)


class SupportAgentState(TypedDict, total=False):
    """State threaded through every node of the orchestrator."""

    # --- Conversation ---
    messages: Annotated[list[BaseMessage], add_messages]
    conversation_id: str

    # --- Identity ---
    customer_id: str
    customer_tier: CustomerTier
    channel: Channel
    # Every channel this conversation has used. The session store is the
    # authority -- it accumulates across turns and survives restarts -- so the
    # ticketing node reads this rather than rebuilding it from ticket state.
    channels: list[str]
    authenticated: bool
    auth_attempts: int

    # --- Classification ---
    intent: str
    intent_confidence: float
    resolution_path: ResolutionPath
    model_tier: ModelTier

    # --- Sentiment ---
    sentiment_score: float
    sentiment_trajectory: list[float]

    # --- Retrieval ---
    rag_context: str
    rag_citations: list[dict[str, Any]]
    rag_hops: int

    # --- Commerce policy gate ---
    # These values are set by Policy Agent and consumed by Order Agent. Keeping
    # them in graph state makes the "policy before action" invariant auditable.
    commerce_order_id: str | None
    return_eligibility: dict[str, Any] | None
    policy_confirmed: bool

    # --- Resolution ---
    draft_response: str
    resolution_status: ResolutionStatus
    tool_attempts: int
    deep_agent_todos: list[str]

    # --- Escalation / ticketing ---
    escalation_required: bool
    escalation_reason: str | None
    handoff_packet: dict[str, Any] | None
    ticket_id: str | None

    # --- Diagnostics (never shown to a customer) ---
    llm_provider: str
    llm_stubbed: bool
    node_timings_ms: Annotated[dict[str, float], merge_timings]


_GREETINGS = (
    "hi",
    "hey",
    "hello",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "cheers",
    "ta",
    "ok",
    "okay",
    "cool",
    "great",
    "bye",
    "goodbye",
    "no thanks",
    "nothing else",
    "that is all",
    "thats all",
    "nope",
    "yes",
    "no",
    "sure",
    "got it",
)


def is_smalltalk(text: str) -> bool:
    """True when a turn carries no actionable request.

    Matters because low classification confidence is a signal to fetch a human
    for an *ambiguous request* -- but a bare greeting is not ambiguous, it just
    has no request in it yet. Escalating "hello" to a human agent is a defect.
    """
    stripped = text.strip().strip(".!,").lower()
    if not stripped:
        return True
    if "?" in text or len(stripped.split()) > 5:
        return False
    return any(
        stripped == phrase or stripped.startswith(f"{phrase} ") or stripped.endswith(f" {phrase}")
        for phrase in _GREETINGS
    )


def new_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex[:16]}"


def initial_state(
    *,
    customer_id: str,
    channel: Channel,
    conversation_id: str | None = None,
    customer_tier: CustomerTier = "standard",
    authenticated: bool = False,
) -> SupportAgentState:
    """Seed a fresh conversation with every field explicitly defaulted."""
    return SupportAgentState(
        messages=[],
        conversation_id=conversation_id or new_conversation_id(),
        customer_id=customer_id,
        customer_tier=customer_tier,
        channel=channel,
        channels=[channel],
        authenticated=authenticated,
        auth_attempts=0,
        intent="general",
        intent_confidence=0.0,
        resolution_path="simple",
        model_tier="slm",
        sentiment_score=0.0,
        sentiment_trajectory=[],
        rag_context="",
        rag_citations=[],
        rag_hops=0,
        commerce_order_id=None,
        return_eligibility=None,
        policy_confirmed=False,
        draft_response="",
        resolution_status="pending",
        tool_attempts=0,
        escalation_required=False,
        escalation_reason=None,
        handoff_packet=None,
        ticket_id=None,
        llm_provider="",
        llm_stubbed=False,
        node_timings_ms={},
    )


def last_user_text(state: SupportAgentState) -> str:
    """Most recent human turn, or empty string."""
    for message in reversed(state.get("messages", [])):
        if message.type == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def transcript(state: SupportAgentState, *, limit: int = 20) -> list[dict[str, str]]:
    """Recent turns as plain dicts -- for prompts, tickets, and handoff packets."""
    rows: list[dict[str, str]] = []
    for message in state.get("messages", [])[-limit:]:
        role = {"human": "customer", "ai": "agent"}.get(message.type, message.type)
        content = message.content
        rows.append({"role": role, "text": content if isinstance(content, str) else str(content)})
    return rows
