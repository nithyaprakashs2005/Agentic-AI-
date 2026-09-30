"""Ticket lifecycle node.

system specification Graph Nodes #8: every interaction creates, updates, or resolves a CRM
ticket enriched with an AI summary, sentiment, intent, and the full transcript.
Runs on both the escalated and the self-resolved paths, so no conversation ends
without a record.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any, Literal

from app.agents.state import SupportAgentState, last_user_text, transcript
from app.services.crm import get_crm
from app.services.sla import evaluate as evaluate_sla
from app.utils.logging import get_logger
from app.utils.pii import scrub

log = get_logger(__name__)

# Fire-and-forget tasks need a strong reference: asyncio only holds a weak one,
# so an unreferenced task can be garbage collected mid-flight and the
# notification would vanish with no error.
_pending_notifications: set[asyncio.Task[None]] = set()


async def ticketing_node(state: SupportAgentState) -> dict[str, Any]:
    started = time.perf_counter()
    crm = get_crm()

    escalated = bool(state.get("escalation_required"))
    packet = state.get("handoff_packet") or {}
    ticket_id = state.get("ticket_id")

    subject = _subject(state)
    status: Literal["open", "pending_human", "resolved"] = (
        "pending_human" if escalated else "resolved"
    )

    ticket = await crm.get_ticket(ticket_id) if ticket_id else None

    if ticket is None:
        # Either a first turn, or the id in the checkpoint points at a ticket
        # that no longer exists. Checkpoints are durable and the ticket store is
        # a separate system, so a dangling reference is possible -- previously
        # the update below silently no-opped and the conversation carried a
        # ghost ticket id for the rest of its life.
        if ticket_id:
            log.warning("ticket.dangling_reference", extra={"ticket_id": ticket_id})
        ticket = await crm.create_ticket(
            customer_id=state.get("customer_id", "unknown"),
            channel=state.get("channel", "webchat"),
            subject=subject,
            intent=state.get("intent", "general"),
            priority=_priority(state, escalated),
            conversation_id=state.get("conversation_id"),
        )
    ticket_id = ticket.ticket_id

    summary = packet.get("executive_summary") or _fallback_summary(state)

    # The SLA clock starts when the customer begins waiting on a human, not
    # when the ticket was opened -- and it is stamped once, so a later turn on
    # the same escalated ticket does not silently reset the deadline.
    escalated_at = ticket.escalated_at
    if escalated and escalated_at is None:
        escalated_at = time.time()

    # A conversation that moves chat -> phone is one ticket spanning two
    # channels. The session store accumulates them across turns, so merge that
    # in rather than relying only on the channel of the current turn.
    channels = list(ticket.channels or [])
    for seen in [*state.get("channels", []), state.get("channel", "webchat")]:
        if seen and seen not in channels:
            channels.append(seen)

    await crm.update_ticket(
        ticket_id,
        summary=summary,
        status=status,
        priority=_priority(state, escalated),
        intent=state.get("intent", "general"),
        sentiment_score=float(state.get("sentiment_score", 0.0)),
        escalated=escalated,
        escalation_reason=state.get("escalation_reason"),
        escalated_at=escalated_at,
        channels=channels,
        citations=state.get("rag_citations", []),
        transcript=[{"role": row["role"], "text": scrub(row["text"])} for row in transcript(state)],
        handoff_packet=packet or None,
    )

    # Tell a human, once, when the ticket first enters the queue. Fire and
    # forget: SMTP must not add seconds to the customer's turn, and a failed
    # notification must not fail it either.
    if escalated and ticket.notified_at is None:
        refreshed = await crm.get_ticket(ticket_id)
        if refreshed is not None:
            task = asyncio.create_task(_notify(crm, refreshed, escalated_at))
            _pending_notifications.add(task)
            task.add_done_callback(_pending_notifications.discard)

    elapsed = round((time.perf_counter() - started) * 1000, 2)
    log.info(
        "ticket.synced",
        extra={
            "ticket_id": ticket_id,
            "status": status,
            "escalated": escalated,
            "latency_ms": elapsed,
        },
    )

    return {
        "ticket_id": ticket_id,
        "resolution_status": "escalated" if escalated else "resolved",
        "node_timings_ms": {"ticketing": elapsed},
    }


async def _notify(crm: Any, ticket: Any, escalated_at: float | None) -> None:
    """Send the handoff alert and record whether it actually went."""
    from app.services.notifications import notify_handoff

    sla = None
    if escalated_at is not None:
        sla = evaluate_sla(
            priority=ticket.priority,
            escalated_at=datetime.fromtimestamp(escalated_at, tz=UTC),
        )

    result = await notify_handoff(ticket=ticket, sla=sla)
    if result.get("sent"):
        # Stamped only on success, so a retry stays possible after an outage.
        await crm.update_ticket(ticket.ticket_id, notified_at=time.time())


def route_after_ticketing(state: SupportAgentState) -> Literal["handoff", "done"]:
    """Escalated conversations park on the HITL gate; the rest terminate."""
    return "handoff" if state.get("escalation_required") else "done"


def _subject(state: SupportAgentState) -> str:
    text = scrub(last_user_text(state)).strip()
    intent = state.get("intent", "general")
    if not text:
        return f"{intent.title()} enquiry"
    condensed = " ".join(text.split())[:120]
    return f"[{intent}] {condensed}"


def _priority(
    state: SupportAgentState, escalated: bool
) -> Literal["low", "normal", "high", "urgent"]:
    tier = state.get("customer_tier", "standard")
    score = float(state.get("sentiment_score", 0.0))

    if escalated and tier == "enterprise":
        return "urgent"
    if escalated or score <= -0.6:
        return "high"
    if score <= -0.3:
        return "normal"
    return "low" if score >= 0.3 else "normal"


def _fallback_summary(state: SupportAgentState) -> str:
    text = scrub(last_user_text(state))
    intent = state.get("intent", "general")
    status = state.get("resolution_status", "pending")
    return f"Customer raised a {intent} enquiry ({status}): {text[:200]}"
