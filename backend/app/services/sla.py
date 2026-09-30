"""Service-level targets for escalated conversations.

A handoff queue without deadlines is just a list. What a support lead needs to
know at a glance is which ticket is about to breach, which is why every escalated
ticket gets a first-response target from the moment it was escalated -- not from
when it was created, because the customer only starts waiting on a human at
handoff.

Targets are derived from priority, which the ticketing node already sets from
tier and sentiment. They are deliberately configurable: every organisation's
support contract differs, and hardcoding "15 minutes" would be a guess dressed
as a policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from app.config import Settings, get_settings

Priority = Literal["low", "normal", "high", "urgent"]
BreachState = Literal["on_track", "due_soon", "breached"]

# Fraction of the window remaining below which a ticket is flagged "due soon".
DUE_SOON_FRACTION = 0.25


@dataclass(slots=True, frozen=True)
class SlaStatus:
    """Where one ticket stands against its first-response target."""

    priority: Priority
    target_minutes: int
    escalated_at: datetime
    due_at: datetime
    remaining_seconds: int
    state: BreachState

    @property
    def overdue_seconds(self) -> int:
        return max(0, -self.remaining_seconds)

    def public(self) -> dict[str, Any]:
        return {
            "priority": self.priority,
            "target_minutes": self.target_minutes,
            "escalated_at": self.escalated_at.isoformat(),
            "due_at": self.due_at.isoformat(),
            "remaining_seconds": self.remaining_seconds,
            "overdue_seconds": self.overdue_seconds,
            "state": self.state,
            "label": self.label(),
        }

    def label(self) -> str:
        """Human phrasing for the queue -- "12m left" or "overdue by 3m"."""
        if self.remaining_seconds >= 0:
            return f"{_duration(self.remaining_seconds)} left"
        return f"overdue by {_duration(self.overdue_seconds)}"


def target_minutes(priority: str, settings: Settings | None = None) -> int:
    """First-response target for a priority, in minutes."""
    settings = settings or get_settings()
    return {
        "urgent": settings.sla_urgent_minutes,
        "high": settings.sla_high_minutes,
        "normal": settings.sla_normal_minutes,
        "low": settings.sla_low_minutes,
    }.get(priority, settings.sla_normal_minutes)


def evaluate(
    *,
    priority: str,
    escalated_at: datetime,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> SlaStatus:
    """Compute the target, deadline, and remaining time for a ticket."""
    settings = settings or get_settings()
    now = now or datetime.now(UTC)

    # A naive timestamp from the database would make the subtraction below
    # raise; treat it as UTC, which is what everything here writes.
    if escalated_at.tzinfo is None:
        escalated_at = escalated_at.replace(tzinfo=UTC)

    minutes = target_minutes(priority, settings)
    due_at = escalated_at + timedelta(minutes=minutes)
    remaining = int((due_at - now).total_seconds())

    if remaining < 0:
        state: BreachState = "breached"
    elif remaining <= minutes * 60 * DUE_SOON_FRACTION:
        state = "due_soon"
    else:
        state = "on_track"

    return SlaStatus(
        priority=priority,  # type: ignore[arg-type]
        target_minutes=minutes,
        escalated_at=escalated_at,
        due_at=due_at,
        remaining_seconds=remaining,
        state=state,
    )


def _duration(seconds: int) -> str:
    """Compact duration: 45s, 12m, 2h 5m, 1d 3h."""
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h" if hours else f"{days}d"


def summarise(statuses: list[SlaStatus]) -> dict[str, Any]:
    """Queue-level counts, so a lead sees the shape of the backlog at a glance."""
    return {
        "total": len(statuses),
        "breached": sum(1 for s in statuses if s.state == "breached"),
        "due_soon": sum(1 for s in statuses if s.state == "due_soon"),
        "on_track": sum(1 for s in statuses if s.state == "on_track"),
        "next_due_seconds": min(
            (s.remaining_seconds for s in statuses if s.remaining_seconds >= 0),
            default=None,
        ),
    }
