"""Ticket endpoints for the agent console and handoff queue."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.crm import get_crm
from app.services.sla import SlaStatus, evaluate, summarise
from app.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/tickets", tags=["tickets"])


class AssignRequest(BaseModel):
    """Claim a ticket so two agents do not work the same handoff."""

    agent: str = Field(min_length=1, max_length=120)


@router.get("")
async def list_tickets(
    limit: int = Query(default=50, ge=1, le=200),
    status: str | None = Query(default=None, description="open | pending_human | resolved"),
) -> dict[str, Any]:
    tickets = await get_crm().list_tickets(limit=limit)
    rows = [_enrich(t) for t in tickets]
    if status:
        rows = [r for r in rows if r["status"] == status]
    return {"count": len(rows), "tickets": rows}


@router.get("/queue")
async def handoff_queue(
    limit: int = Query(default=50, ge=1, le=200),
    sort: str = Query(default="due", description="due | newest"),
) -> dict[str, Any]:
    """Conversations waiting on a human, with their response deadlines.

    Sorted by deadline rather than recency by default: what a support lead needs
    first is whatever breaches soonest, which is not the newest ticket.
    """
    tickets = await get_crm().list_tickets(limit=limit)
    queued = [t for t in tickets if t.status == "pending_human"]

    entries = []
    statuses: list[SlaStatus] = []
    for ticket in queued:
        row = _enrich(ticket)
        row["handoff_packet"] = ticket.handoff_packet
        entries.append(row)
        if row["sla"]:
            statuses.append(_sla_for(ticket))  # type: ignore[arg-type]

    if sort == "due":
        # Tickets with no deadline sort last rather than first.
        entries.sort(key=lambda e: (e["sla"] or {}).get("remaining_seconds", 1 << 30))
    else:
        entries.sort(key=lambda e: -(e["updated_at"] or 0))

    return {"count": len(entries), "queue": entries, "sla_summary": summarise(statuses)}


@router.post("/{ticket_id}/assign")
async def assign_ticket(ticket_id: str, payload: AssignRequest) -> dict[str, Any]:
    """Claim a ticket, and stamp first response if this is the first touch."""
    crm = get_crm()
    ticket = await crm.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")

    fields: dict[str, Any] = {"assigned_to": payload.agent}
    if ticket.first_responded_at is None:
        # First human touch stops the first-response clock.
        fields["first_responded_at"] = datetime.now(UTC).timestamp()

    updated = await crm.update_ticket(ticket_id, **fields)
    log.info("ticket.assigned", extra={"ticket_id": ticket_id, "agent": payload.agent})
    return _enrich(updated) if updated else {"ticket_id": ticket_id}


@router.get("/{ticket_id}")
async def get_ticket(ticket_id: str) -> dict[str, Any]:
    ticket = await get_crm().get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return {
        **_enrich(ticket),
        "transcript": ticket.transcript,
        "handoff_packet": ticket.handoff_packet,
    }


def _sla_for(ticket: Any) -> SlaStatus | None:
    """SLA only applies once a human is actually waiting to be reached."""
    if not ticket.escalated or ticket.escalated_at is None:
        return None
    return evaluate(
        priority=ticket.priority,
        escalated_at=datetime.fromtimestamp(ticket.escalated_at, tz=UTC),
    )


def _enrich(ticket: Any) -> dict[str, Any]:
    """Ticket plus derived timing the console needs to prioritise work."""
    row = dict(ticket.public())
    sla = _sla_for(ticket)
    row["sla"] = sla.public() if sla else None

    now = datetime.now(UTC).timestamp()
    row["age_seconds"] = int(now - (ticket.created_at or now))
    row["waiting_seconds"] = int(now - ticket.escalated_at) if ticket.escalated_at else None
    # ISO alongside the epoch: the epoch is for sorting, the ISO for display
    # without every client reimplementing the conversion.
    row["created_at_iso"] = _iso(ticket.created_at)
    row["updated_at_iso"] = _iso(ticket.updated_at)
    row["escalated_at_iso"] = _iso(ticket.escalated_at)
    row["first_responded_at_iso"] = _iso(ticket.first_responded_at)
    return row


def _iso(epoch: float | None) -> str | None:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat() if epoch else None
