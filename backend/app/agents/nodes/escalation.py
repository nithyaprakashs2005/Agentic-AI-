"""Escalation: handoff packet generation and the HITL interrupt.

system specification Graph Nodes #7. Split across two nodes on purpose:

* `escalation_node` builds the packet and tells the customer a human is coming.
* `human_handoff_node` raises the LangGraph `interrupt()`.

The split matters because `interrupt()` unwinds the node -- state written in the
same node before the pause is discarded and recomputed on resume. Putting the
pause in its own terminal node lets the ticket be created *first*, so the human
picking the conversation up already has a CRM record waiting.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app.agents.nodes.sentiment import evaluate_escalation
from app.agents.prompts.voice import VOICE_ESCALATION_PHRASE
from app.agents.state import SupportAgentState, last_user_text, transcript
from app.services.llm import get_gateway
from app.utils.logging import get_logger
from app.utils.pii import scrub

log = get_logger(__name__)

_ACK_TEXT = (
    "I am bringing in a colleague who can take this further for you. "
    "They will have the full history of our conversation, so you will not need "
    "to repeat anything."
)

_ACK_AUTH_FAILED = (
    "I was not able to verify the account details, so I am connecting you with a "
    "colleague who can help confirm your identity securely."
)


async def escalation_node(state: SupportAgentState) -> dict[str, Any]:
    """Assemble the structured handoff packet and acknowledge to the customer."""
    started = time.perf_counter()

    _, reason = evaluate_escalation(state)
    reason = reason or "unspecified"

    packet = await _build_handoff_packet(state, reason)

    ack = _ACK_AUTH_FAILED if reason == "identity_verification_failed" else _ACK_TEXT
    if state.get("channel") == "voice":
        ack = VOICE_ESCALATION_PHRASE

    elapsed = round((time.perf_counter() - started) * 1000, 2)
    log.info(
        "escalation.packet_built",
        extra={
            "reason": reason,
            "churn_risk": packet["churn_risk"],
            "citations": len(packet["rag_citations"]),
            "latency_ms": elapsed,
        },
    )

    return {
        "escalation_required": True,
        "escalation_reason": reason,
        "resolution_status": "escalated",
        "handoff_packet": packet,
        "draft_response": ack,
        "messages": [AIMessage(content=ack)],
        "node_timings_ms": {"escalation": elapsed},
    }


def human_handoff_node(state: SupportAgentState) -> dict[str, Any]:
    """Terminal HITL gate.

    Raises a dynamic `interrupt()` carrying the handoff packet. The graph stays
    parked on its checkpoint until a human agent resumes it with
    `Command(resume={...})`.
    """
    packet = state.get("handoff_packet") or {}
    resolution = interrupt(
        {
            "type": "human_handoff",
            "ticket_id": state.get("ticket_id"),
            "reason": state.get("escalation_reason"),
            "packet": packet,
        }
    )
    # Reached only after a human resumes the run.
    log.info(
        "escalation.resumed",
        extra={"ticket_id": state.get("ticket_id"), "outcome": str(resolution)[:200]},
    )
    return {
        "resolution_status": "resolved",
        "escalation_required": False,
    }


async def _build_handoff_packet(state: SupportAgentState, reason: str) -> dict[str, Any]:
    """Structured context packet. Transcript and summary are PII-scrubbed."""
    trajectory = [float(v) for v in state.get("sentiment_trajectory", [])]
    summary = await _summarise(state)
    actions = _actions_attempted(state)
    history = [{"role": row["role"], "text": scrub(row["text"])} for row in transcript(state)]

    return {
        "ticket_id": state.get("ticket_id"),
        "conversation_id": state.get("conversation_id"),
        "customer_id": state.get("customer_id"),
        "customer_tier": state.get("customer_tier", "standard"),
        "channel": state.get("channel"),
        "escalation_reason": reason,
        "executive_summary": summary,
        "intent": state.get("intent"),
        "intent_confidence": state.get("intent_confidence"),
        # Stable, plainly named fields make the packet consumable by a human
        # queue as well as the richer CRM ticket model.
        "issue": scrub(last_user_text(state)),
        "history": history,
        "actions_taken": actions,
        "actions_attempted": actions,
        "rag_citations": state.get("rag_citations", []),
        "sentiment_trajectory": trajectory,
        "churn_risk": _churn_risk(state, reason, trajectory),
        "resolution_path_used": state.get("resolution_path", "simple"),
        "deep_agent_todos": state.get("deep_agent_todos", []),
        "recommended_actions": _recommended_actions(state, reason),
        "transcript": history,
        "authenticated": bool(state.get("authenticated")),
    }


async def _summarise(state: SupportAgentState) -> str:
    from app.agents.prompts.system import TICKET_SUMMARY_PROMPT

    rows = transcript(state)
    if not rows:
        return "No conversation content."

    body = "\n".join(f"{row['role']}: {scrub(row['text'])}" for row in rows)
    result = await get_gateway().complete(
        [
            {"role": "system", "content": TICKET_SUMMARY_PROMPT},
            {"role": "user", "content": body[:4000]},
        ],
        tier="frontier",
        purpose="summary",
        temperature=0.2,
        max_tokens=600,  # headroom for reasoning tokens
    )
    return result.text.strip()


def _actions_attempted(state: SupportAgentState) -> list[str]:
    actions: list[str] = []
    hops = int(state.get("rag_hops", 0))
    if hops:
        citations = len(state.get("rag_citations", []))
        actions.append(f"Knowledge base search over {hops} hop(s), {citations} verified source(s)")
    attempts = int(state.get("tool_attempts", 0))
    if attempts:
        actions.append(f"{attempts} CRM/billing lookup(s) via MCP tools")
    for todo in state.get("deep_agent_todos", []):
        actions.append(f"Investigation step: {todo}")
    if not state.get("authenticated"):
        actions.append("Identity verification attempted and not completed")
    return actions or ["No automated resolution steps completed"]


def _churn_risk(state: SupportAgentState, reason: str, trajectory: list[float]) -> str:
    tier = state.get("customer_tier", "standard")
    worst = min(trajectory) if trajectory else 0.0
    falling = len(trajectory) >= 2 and trajectory[-1] < trajectory[0]

    if reason.startswith("explicit_keyword:cancel") or "delete_my_account" in reason:
        return "high"
    if worst <= -0.6 and (tier == "enterprise" or falling):
        return "high"
    if worst <= -0.3:
        return "medium"
    return "low"


def _recommended_actions(state: SupportAgentState, reason: str) -> list[str]:
    intent = state.get("intent", "general")
    tier = state.get("customer_tier", "standard")
    recommended: list[str] = []

    if reason == "identity_verification_failed":
        recommended.append("Verify identity through an out-of-band channel before disclosing data")
    if reason == "low_intent_confidence":
        recommended.append("Re-read the opening message; automated classification was unreliable")
    if reason == "tool_attempts_exhausted":
        recommended.append("Check whether the CRM or billing systems are degraded")
    if reason == "repeated_complaint":
        recommended.append("Review prior fulfilment failures and offer a durable resolution")
    if reason == "refund_timing_exception":
        recommended.append(
            "Trace the refund with the payment provider; the stated maximum window has passed"
        )
    if reason.startswith("explicit_keyword"):
        recommended.append("Customer asked for a human directly; lead by acknowledging that")

    match intent:
        case "billing":
            recommended.append("Review the invoice history for duplicate authorisations")
            recommended.append("Confirm whether a goodwill credit is warranted under policy")
        case "returns":
            recommended.append("Confirm return eligibility against the 30 day delivery window")
        case "orders":
            recommended.append("Check carrier tracking for a stalled or misrouted parcel")
        case "technical":
            recommended.append("Check the status page for a known incident in this window")

    if tier == "enterprise":
        recommended.append("Enterprise account: loop in the named account manager")

    last = last_user_text(state)
    if last and "?" in last:
        recommended.append("The customer's last message ended in an unanswered question")

    return recommended or ["Review the transcript and continue where the agent stopped"]
