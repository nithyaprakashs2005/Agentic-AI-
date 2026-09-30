"""Outbound notification for handoffs.

An escalation that only appears in a queue nobody is watching is not a handoff.
When a conversation is passed to a human, the support address is emailed with
enough context to act on before opening the console.

The send is real SMTP. With no host configured it reports `sent: False` and says
why -- it never claims delivery it did not achieve, because a support lead acting
on "the team was notified" when nothing was sent is worse than knowing the
channel is down.

Sending is deliberately off the request path: SMTP can block for seconds, and a
customer's turn must not wait on it.
"""

from __future__ import annotations

import asyncio
import smtplib
from datetime import UTC, datetime
from email.message import EmailMessage
from typing import Any

from app.config import Settings, get_settings
from app.services.sla import SlaStatus
from app.utils.logging import get_logger

log = get_logger(__name__)

SEND_TIMEOUT_SECONDS = 15.0


def build_handoff_email(
    *,
    ticket: Any,
    sla: SlaStatus | None,
    console_url: str,
    settings: Settings,
) -> EmailMessage:
    """Compose the alert. Plain text: it has to be readable on a phone at 2am."""
    packet = getattr(ticket, "handoff_packet", None) or {}
    priority = getattr(ticket, "priority", "normal")
    tier = packet.get("customer_tier", "standard")

    urgency = "URGENT" if priority == "urgent" else priority.upper()
    subject = f"[{urgency}] Handoff {ticket.ticket_id} - {getattr(ticket, 'subject', '')[:70]}"

    lines = [
        "A conversation has been handed to a human.",
        "",
        f"Ticket      {ticket.ticket_id}",
        f"Priority    {priority}" + (f"  (tier: {tier})" if tier else ""),
        f"Channel     {_channels(ticket)}",
        f"Intent      {getattr(ticket, 'intent', 'general')}",
        f"Reason      {packet.get('escalation_reason', 'unspecified')}",
    ]

    if sla is not None:
        lines += [
            f"Escalated   {sla.escalated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Respond by  {sla.due_at.strftime('%Y-%m-%d %H:%M UTC')}  ({sla.label()})",
        ]

    if packet.get("churn_risk"):
        lines.append(f"Churn risk  {packet['churn_risk']}")

    summary = packet.get("executive_summary") or getattr(ticket, "summary", "")
    if summary:
        lines += ["", "Summary", f"  {summary}"]

    attempted = packet.get("actions_attempted") or []
    if attempted:
        lines += ["", "Already attempted"]
        lines += [f"  - {a}" for a in attempted[:6]]

    recommended = packet.get("recommended_actions") or []
    if recommended:
        lines += ["", "Recommended next"]
        lines += [f"  - {a}" for a in recommended[:6]]

    lines += ["", f"Open it: {console_url}", ""]
    # The transcript is PII-scrubbed upstream, but it still travels by email, so
    # only the tail goes -- enough to pick up the thread, not the whole history.
    transcript = packet.get("transcript") or []
    if transcript:
        lines += ["Last messages"]
        for row in transcript[-4:]:
            lines.append(f"  {row.get('role', '?')}: {row.get('text', '')[:160]}")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.smtp_from_address or "support@localhost"
    message["To"] = ", ".join(settings.handoff_notify_list)
    message.set_content("\n".join(lines))
    return message


def _channels(ticket: Any) -> str:
    channels = getattr(ticket, "channels", None) or []
    if channels:
        return " -> ".join(channels)
    return str(getattr(ticket, "channel", "unknown"))


def _send_blocking(message: EmailMessage, settings: Settings) -> None:
    """Synchronous SMTP, run off the event loop by the caller."""
    if settings.smtp_use_tls:
        server = smtplib.SMTP(
            settings.smtp_host or "", settings.smtp_port, timeout=SEND_TIMEOUT_SECONDS
        )
        server.starttls()
    else:
        server = smtplib.SMTP(
            settings.smtp_host or "", settings.smtp_port, timeout=SEND_TIMEOUT_SECONDS
        )

    with server:
        if settings.smtp_username and settings.smtp_password:
            server.login(settings.smtp_username, settings.smtp_password.get_secret_value())
        server.send_message(message)


async def notify_handoff(
    *,
    ticket: Any,
    sla: SlaStatus | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Email the support queue about a handoff.

    Returns a result dict rather than raising: a failed notification must not
    fail the customer's turn. `sent` is only ever true when SMTP accepted the
    message.
    """
    settings = settings or get_settings()

    if not settings.smtp_ready:
        missing = []
        if not settings.smtp_host:
            missing.append("SMTP_HOST")
        if not settings.smtp_from_address:
            missing.append("SMTP_FROM_ADDRESS")
        if not settings.handoff_notify_list:
            missing.append("HANDOFF_NOTIFY_EMAILS")
        log.info(
            "handoff.notification_skipped",
            extra={"ticket_id": ticket.ticket_id, "missing": missing},
        )
        return {"sent": False, "reason": "smtp_not_configured", "missing": missing}

    console_url = f"{settings.cors_origin_list[0] if settings.cors_origin_list else ''}/dashboard"
    message = build_handoff_email(
        ticket=ticket, sla=sla, console_url=console_url, settings=settings
    )

    try:
        # SMTP blocks; keep it off the loop so a slow mail server cannot stall
        # unrelated conversations.
        await asyncio.wait_for(
            asyncio.to_thread(_send_blocking, message, settings),
            timeout=SEND_TIMEOUT_SECONDS + 5,
        )
    except TimeoutError:
        log.warning("handoff.notification_timeout", extra={"ticket_id": ticket.ticket_id})
        return {"sent": False, "reason": "timeout"}
    except Exception as exc:
        log.warning(
            "handoff.notification_failed",
            extra={"ticket_id": ticket.ticket_id, "error": str(exc)[:200]},
        )
        return {"sent": False, "reason": "send_failed"}

    log.info(
        "handoff.notified",
        extra={
            "ticket_id": ticket.ticket_id,
            "recipients": len(settings.handoff_notify_list),
        },
    )
    return {
        "sent": True,
        "recipients": len(settings.handoff_notify_list),
        "at": datetime.now(UTC).isoformat(),
    }
