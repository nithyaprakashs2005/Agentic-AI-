"""Handoff SLA, channel accumulation, assignment, and notification.

These cover what a support lead actually needs from a queue: how long is left,
which conversation breaches first, where the customer has been talking, and
whether anyone was told.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.config import Settings
from app.services.sla import _duration, evaluate, summarise, target_minutes

pytestmark = pytest.mark.timeout(120)


def at(minutes_ago: float) -> datetime:
    return datetime.now(UTC) - timedelta(minutes=minutes_ago)


# --- Targets ------------------------------------------------------------
@pytest.mark.parametrize(
    ("priority", "expected"),
    [("urgent", 15), ("high", 60), ("normal", 240), ("low", 480)],
)
def test_targets_by_priority(priority: str, expected: int, _offline_settings: Settings) -> None:
    assert target_minutes(priority, _offline_settings) == expected


def test_unknown_priority_falls_back_to_normal(_offline_settings: Settings) -> None:
    assert target_minutes("nonsense", _offline_settings) == _offline_settings.sla_normal_minutes


def test_targets_come_from_settings() -> None:
    """A support contract is configuration, not a constant in the code."""
    strict = Settings(_env_file=None, sla_urgent_minutes=5)  # type: ignore[call-arg]
    assert target_minutes("urgent", strict) == 5


# --- Remaining time and state ------------------------------------------
def test_fresh_urgent_ticket_is_on_track(_offline_settings: Settings) -> None:
    status = evaluate(priority="urgent", escalated_at=at(1), settings=_offline_settings)
    assert status.state == "on_track"
    assert 13 * 60 <= status.remaining_seconds <= 15 * 60
    assert status.label().endswith("left")


def test_ticket_past_three_quarters_is_due_soon(_offline_settings: Settings) -> None:
    """Warn before the breach, not at it -- 15m target, 12m elapsed."""
    status = evaluate(priority="urgent", escalated_at=at(12), settings=_offline_settings)
    assert status.state == "due_soon"


def test_ticket_past_its_target_is_breached(_offline_settings: Settings) -> None:
    status = evaluate(priority="urgent", escalated_at=at(20), settings=_offline_settings)
    assert status.state == "breached"
    assert status.remaining_seconds < 0
    assert status.overdue_seconds > 0
    assert "overdue by" in status.label()


def test_due_at_is_escalation_plus_target(_offline_settings: Settings) -> None:
    """The clock starts at handoff, not at ticket creation."""
    escalated = at(0)
    status = evaluate(priority="high", escalated_at=escalated, settings=_offline_settings)
    assert status.due_at == escalated + timedelta(minutes=60)


def test_naive_timestamps_are_treated_as_utc(_offline_settings: Settings) -> None:
    """A naive datetime out of the database must not raise on subtraction."""
    naive = datetime.now(UTC).replace(tzinfo=None)
    status = evaluate(priority="normal", escalated_at=naive, settings=_offline_settings)
    assert status.escalated_at.tzinfo is not None


# --- Summary ------------------------------------------------------------
def test_summary_counts_each_state(_offline_settings: Settings) -> None:
    statuses = [
        evaluate(priority="urgent", escalated_at=at(1), settings=_offline_settings),
        evaluate(priority="urgent", escalated_at=at(13), settings=_offline_settings),
        evaluate(priority="urgent", escalated_at=at(30), settings=_offline_settings),
    ]
    summary = summarise(statuses)
    assert summary == {
        "total": 3,
        "breached": 1,
        "due_soon": 1,
        "on_track": 1,
        "next_due_seconds": pytest.approx(summary["next_due_seconds"]),
    }


def test_summary_next_due_ignores_breached(_offline_settings: Settings) -> None:
    """ "Next due" means the next deadline coming, not one already missed."""
    statuses = [
        evaluate(priority="urgent", escalated_at=at(30), settings=_offline_settings),
        evaluate(priority="high", escalated_at=at(1), settings=_offline_settings),
    ]
    assert summarise(statuses)["next_due_seconds"] > 0


def test_summary_of_an_empty_queue(_offline_settings: Settings) -> None:
    assert summarise([])["next_due_seconds"] is None


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(45, "45s"), (90, "1m"), (3600, "1h"), (7500, "2h 5m"), (90000, "1d 1h")],
)
def test_duration_formatting(seconds: int, expected: str) -> None:
    assert _duration(seconds) == expected


# --- End to end through the API ----------------------------------------
async def test_escalated_ticket_carries_a_deadline(api_client: Any) -> None:
    await api_client.post(
        "/api/chat",
        json={"message": "I want to speak to a human", "conversation_id": "sla-1"},
    )
    body = (await api_client.get("/api/tickets/queue")).json()
    assert body["queue"], "escalation did not reach the queue"

    entry = body["queue"][0]
    assert entry["sla"] is not None
    assert entry["sla"]["remaining_seconds"] > 0
    assert entry["escalated_at_iso"]
    assert entry["created_at_iso"]
    assert entry["waiting_seconds"] is not None


async def test_queue_is_ordered_by_deadline(api_client: Any) -> None:
    """A lead needs whatever breaches soonest first, not the newest ticket."""
    await api_client.post(
        "/api/chat",
        json={"message": "I want a human", "conversation_id": "sla-low", "customer_tier": "free"},
    )
    await api_client.post(
        "/api/chat",
        json={
            "message": "I want a human",
            "conversation_id": "sla-urgent",
            "customer_id": "cust_abc123",
            "customer_tier": "enterprise",
        },
    )

    queue = (await api_client.get("/api/tickets/queue")).json()["queue"]
    remaining = [e["sla"]["remaining_seconds"] for e in queue if e["sla"]]
    assert remaining == sorted(remaining), "queue is not ordered by deadline"


async def test_channels_accumulate_across_a_conversation(api_client: Any) -> None:
    """Chat then phone is one ticket spanning two channels, not two tickets."""
    first = await api_client.post(
        "/api/chat",
        json={
            "message": "How long do refunds take?",
            "channel": "webchat",
            "conversation_id": "ch-1",
        },
    )
    ticket_id = first.json()["ticket_id"]

    await api_client.post(
        "/api/chat",
        json={"message": "and bank transfers?", "channel": "voice", "conversation_id": "ch-1"},
    )

    body = (await api_client.get(f"/api/tickets/{ticket_id}")).json()
    assert "webchat" in body["channels"]
    assert "voice" in body["channels"]


async def test_claiming_a_ticket_stops_the_first_response_clock(api_client: Any) -> None:
    chat = await api_client.post(
        "/api/chat", json={"message": "get me a manager", "conversation_id": "assign-1"}
    )
    ticket_id = chat.json()["ticket_id"]

    before = (await api_client.get(f"/api/tickets/{ticket_id}")).json()
    assert before["assigned_to"] is None
    assert before["first_responded_at_iso"] is None

    claimed = (
        await api_client.post(f"/api/tickets/{ticket_id}/assign", json={"agent": "dana"})
    ).json()
    assert claimed["assigned_to"] == "dana"
    assert claimed["first_responded_at_iso"] is not None


async def test_claiming_twice_keeps_the_original_response_time(api_client: Any) -> None:
    """Reassigning must not rewrite when the customer was first reached."""
    chat = await api_client.post(
        "/api/chat", json={"message": "get me a manager", "conversation_id": "assign-2"}
    )
    ticket_id = chat.json()["ticket_id"]

    first = (
        await api_client.post(f"/api/tickets/{ticket_id}/assign", json={"agent": "dana"})
    ).json()
    second = (
        await api_client.post(f"/api/tickets/{ticket_id}/assign", json={"agent": "sam"})
    ).json()

    assert second["assigned_to"] == "sam"
    assert second["first_responded_at_iso"] == first["first_responded_at_iso"]


async def test_assigning_an_unknown_ticket_is_404(api_client: Any) -> None:
    resp = await api_client.post("/api/tickets/TKT-nope/assign", json={"agent": "dana"})
    assert resp.status_code == 404


async def test_unescalated_tickets_have_no_deadline(api_client: Any) -> None:
    """An SLA only means something once a human is actually being waited on."""
    chat = await api_client.post(
        "/api/chat", json={"message": "hello", "conversation_id": "no-sla"}
    )
    body = (await api_client.get(f"/api/tickets/{chat.json()['ticket_id']}")).json()
    if not body["escalated"]:
        assert body["sla"] is None
        assert body["waiting_seconds"] is None
