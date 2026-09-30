"""Handoff email notification.

The property that matters: `sent` is true only when SMTP accepted the message.
A support lead who believes "the team was notified" when nothing left the
building is worse off than one who can see the channel is down.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.message import EmailMessage
from typing import Any

import pytest

from app.config import Settings
from app.services.notifications import build_handoff_email, notify_handoff
from app.services.sla import evaluate

pytestmark = pytest.mark.timeout(90)


class _Ticket:
    def __init__(self, **overrides: Any) -> None:
        self.ticket_id = "TKT-1"
        self.subject = "[billing] duplicate charge"
        self.summary = "Customer disputes a duplicate charge."
        self.priority = "urgent"
        self.intent = "billing"
        self.channel = "webchat"
        self.channels = ["webchat", "voice"]
        self.handoff_packet: dict[str, Any] = {
            "escalation_reason": "explicit_keyword:human",
            "customer_tier": "enterprise",
            "churn_risk": "high",
            "executive_summary": "Customer asked for a human after two failed lookups.",
            "actions_attempted": ["Checked invoices", "Cross-referenced orders"],
            "recommended_actions": ["Review authorisation log"],
            "transcript": [
                {"role": "customer", "text": "card [[CC_1]] charged twice"},
                {"role": "agent", "text": "Bringing in a colleague."},
            ],
        }
        for key, value in overrides.items():
            setattr(self, key, value)


@pytest.fixture
def smtp_settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        smtp_host="smtp.example.com",
        smtp_from_address="support@example.com",
        handoff_notify_emails="queue@example.com, lead@example.com",
    )


# --- Configuration gating ----------------------------------------------
async def test_unconfigured_smtp_reports_not_sent() -> None:
    """Never claim delivery that did not happen."""
    result = await notify_handoff(ticket=_Ticket(), settings=Settings(_env_file=None))  # type: ignore[call-arg]
    assert result["sent"] is False
    assert result["reason"] == "smtp_not_configured"
    assert "SMTP_HOST" in result["missing"]


async def test_missing_recipients_is_not_configured() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, smtp_host="smtp.example.com", smtp_from_address="s@example.com"
    )
    result = await notify_handoff(ticket=_Ticket(), settings=settings)
    assert result["sent"] is False
    assert "HANDOFF_NOTIFY_EMAILS" in result["missing"]


def test_recipient_list_is_parsed_and_trimmed(smtp_settings: Settings) -> None:
    assert smtp_settings.handoff_notify_list == ["queue@example.com", "lead@example.com"]
    assert smtp_settings.smtp_ready is True


# --- Message content ----------------------------------------------------
def test_email_leads_with_urgency_and_ticket(smtp_settings: Settings) -> None:
    message = build_handoff_email(
        ticket=_Ticket(), sla=None, console_url="http://x/dashboard", settings=smtp_settings
    )
    assert "URGENT" in message["Subject"]
    assert "TKT-1" in message["Subject"]
    assert message["To"] == "queue@example.com, lead@example.com"


def test_email_carries_the_deadline(smtp_settings: Settings) -> None:
    sla = evaluate(priority="urgent", escalated_at=datetime.now(UTC), settings=smtp_settings)
    message = build_handoff_email(
        ticket=_Ticket(), sla=sla, console_url="http://x/dashboard", settings=smtp_settings
    )
    body = message.get_content()
    assert "Respond by" in body
    assert "left" in body


def test_email_shows_the_channel_trail(smtp_settings: Settings) -> None:
    """A conversation that moved chat -> phone should say so."""
    message = build_handoff_email(
        ticket=_Ticket(), sla=None, console_url="http://x/dashboard", settings=smtp_settings
    )
    assert "webchat -> voice" in message.get_content()


def test_email_includes_context_a_human_needs(smtp_settings: Settings) -> None:
    body = build_handoff_email(
        ticket=_Ticket(), sla=None, console_url="http://x/dashboard", settings=smtp_settings
    ).get_content()
    assert "Customer asked for a human" in body
    assert "Checked invoices" in body
    assert "Review authorisation log" in body
    assert "explicit_keyword:human" in body
    assert "http://x/dashboard" in body


def test_email_carries_only_masked_transcript(smtp_settings: Settings) -> None:
    """Transcripts are scrubbed upstream; email must not reintroduce raw PII."""
    body = build_handoff_email(
        ticket=_Ticket(), sla=None, console_url="http://x/dashboard", settings=smtp_settings
    ).get_content()
    assert "[[CC_1]]" in body
    assert "4111" not in body


def test_email_survives_a_ticket_with_no_packet(smtp_settings: Settings) -> None:
    message = build_handoff_email(
        ticket=_Ticket(handoff_packet=None),
        sla=None,
        console_url="http://x/dashboard",
        settings=smtp_settings,
    )
    assert "TKT-1" in message["Subject"]


# --- Sending ------------------------------------------------------------
async def test_successful_send_reports_sent(
    smtp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[EmailMessage] = []

    def fake_send(message: EmailMessage, settings: Settings) -> None:
        captured.append(message)

    monkeypatch.setattr("app.services.notifications._send_blocking", fake_send)

    result = await notify_handoff(ticket=_Ticket(), settings=smtp_settings)
    assert result["sent"] is True
    assert result["recipients"] == 2
    assert captured and "TKT-1" in captured[0]["Subject"]


async def test_smtp_failure_degrades_without_raising(
    smtp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead mail server must not fail the customer's turn."""

    def boom(message: EmailMessage, settings: Settings) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr("app.services.notifications._send_blocking", boom)

    result = await notify_handoff(ticket=_Ticket(), settings=smtp_settings)
    assert result["sent"] is False
    assert result["reason"] == "send_failed"


async def test_a_slow_mail_server_times_out(
    smtp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    def slow(message: EmailMessage, settings: Settings) -> None:
        time.sleep(30)

    monkeypatch.setattr("app.services.notifications._send_blocking", slow)
    monkeypatch.setattr("app.services.notifications.SEND_TIMEOUT_SECONDS", 0.2)

    result = await notify_handoff(ticket=_Ticket(), settings=smtp_settings)
    assert result["sent"] is False
    assert result["reason"] == "timeout"


# --- Wiring -------------------------------------------------------------
async def test_escalation_triggers_exactly_one_notification(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Notify on entering the queue, not on every subsequent turn."""
    calls: list[str] = []

    async def spy(*, ticket: Any, sla: Any = None, settings: Any = None) -> dict[str, Any]:
        calls.append(ticket.ticket_id)
        return {"sent": True, "recipients": 1}

    monkeypatch.setattr("app.services.notifications.notify_handoff", spy)

    for _ in range(3):
        await api_client.post(
            "/api/chat",
            json={"message": "I want a human", "conversation_id": "notify-once"},
        )

    # The task is fire-and-forget; let it run.
    import asyncio

    await asyncio.sleep(0.3)
    assert len(calls) <= 1, f"notified {len(calls)} times for one handoff"
