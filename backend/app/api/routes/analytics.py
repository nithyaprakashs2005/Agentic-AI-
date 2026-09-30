"""Analytics endpoints: aggregates and CSAT capture.

CSAT exists as an endpoint rather than a derived number because it can only
come from customers. `POST /api/analytics/feedback` is where a rating enters the
system; until one does, the CSAT figure is `null`, not a guess.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.analytics import CsatResponse, get_metrics
from app.services.crm import get_crm
from app.services.repository import get_repository
from app.utils.logging import get_logger
from app.utils.pii import scrub

log = get_logger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_WINDOWS: dict[str, float | None] = {
    "1h": 3600.0,
    "24h": 86_400.0,
    "7d": 604_800.0,
    "all": None,
}


class FeedbackRequest(BaseModel):
    ticket_id: str = Field(max_length=64)
    score: int = Field(ge=1, le=5, description="Customer satisfaction, 1-5")
    comment: str = Field(default="", max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=64)


@router.get("/summary")
async def summary(
    window: Literal["1h", "24h", "7d", "all"] = Query(default="24h"),
) -> dict[str, Any]:
    """Aggregate metrics over a time window."""
    return get_metrics().snapshot(_WINDOWS[window])


@router.get("/health")
async def analytics_health() -> dict[str, Any]:
    """How much data backs the numbers, and how durable it is."""
    described = get_metrics().describe()
    return {
        **described,
        "durable": False,
        "note": "metrics live in-process and reset on restart",
    }


@router.post("/feedback")
async def submit_feedback(payload: FeedbackRequest) -> dict[str, Any]:
    """Record a customer satisfaction rating against a ticket."""
    ticket = await get_crm().get_ticket(payload.ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")

    repo = get_repository()
    durable = False
    if repo is not None:
        # Durable: CSAT must survive a restart, unlike the metrics ring buffer.
        try:
            await repo.save_feedback(
                ticket_id=payload.ticket_id,
                score=payload.score,
                comment=scrub(payload.comment),
                conversation_id=payload.conversation_id,
            )
            durable = True
        except Exception:
            # Keep the rating rather than rejecting the customer's submission,
            # but do not let the response imply it was stored durably.
            log.warning("csat.persist_failed", extra={"ticket_id": payload.ticket_id})

    get_metrics().record_csat(
        CsatResponse(
            ticket_id=payload.ticket_id,
            conversation_id=payload.conversation_id,
            score=payload.score,
            # Free text from a customer can carry anything.
            comment=scrub(payload.comment),
        )
    )
    return {
        "recorded": True,
        "durable": durable,
        "ticket_id": payload.ticket_id,
        "score": payload.score,
    }
