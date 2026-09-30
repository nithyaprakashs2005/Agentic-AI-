"""Durable ticket and feedback storage.

The schema and migrations existed before this module did, but nothing wrote
through them -- tickets lived only in the CRM adapter's dict, so a restart lost
every one. This closes that gap.

Scope is deliberately narrow. Customers, orders, and invoices stay seeded in
`crm.py`: in a real deployment they are *read* from the CRM of record, not owned
here. Tickets and CSAT responses are ours, so they are what gets persisted.

One wrinkle worth knowing: `tickets.customer_id` is a foreign key, and inbound
SMS or email can produce a customer we have never seen (`anon:+1555...`). Rather
than dropping the constraint, `_ensure_customer` upserts a row first, so the
foreign key keeps its meaning and unknown contacts still get a ticket.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Concatenate, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings, get_settings
from app.models.tables import Conversation, Customer, FeedbackResponse, Message, Ticket
from app.utils.logging import get_logger

log = get_logger(__name__)


class RepositoryTimeoutError(Exception):
    """The database did not answer inside its budget.

    Deliberately a plain `Exception` so the existing callers -- which already
    fall back to the in-memory store on failure -- engage without needing to
    know this type exists.
    """


def _bounded[**P, R](
    method: Callable[Concatenate[TicketRepository, P], Awaitable[R]],
) -> Callable[Concatenate[TicketRepository, P], Awaitable[R]]:
    """Give a repository call a deadline.

    Without one, an unreachable database does not raise -- it *stalls*. A
    connection pooled while Postgres was healthy keeps a socket whose peer has
    since vanished, and the first query on it waits out the OS TCP retransmit
    before failing. Measured here: `/api/tickets/queue` returned 200 after 260
    seconds, having fallen back correctly but four minutes too late to matter.

    The fallback logic was never wrong; it was simply unreachable in the time
    anyone was willing to wait. This makes the failure prompt so it can run.
    """

    @functools.wraps(method)
    async def wrapper(self: TicketRepository, *args: P.args, **kwargs: P.kwargs) -> R:
        # A store known to be down is not worth waiting on again. Without this,
        # every caller pays the full budget on every call -- a dashboard polling
        # a dead database every 10s would stall 5s each time, forever.
        if self._breaker_open():
            raise RepositoryTimeoutError(f"{method.__name__} (circuit open)")
        try:
            async with asyncio.timeout(self._timeout):
                result = await method(self, *args, **kwargs)
        except TimeoutError as exc:
            self._record_failure()
            log.warning(
                "repository.timeout",
                extra={"operation": method.__name__, "budget_seconds": self._timeout},
            )
            raise RepositoryTimeoutError(method.__name__) from exc
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result

    # functools.wraps erases the positional-vs-named shape of `self`; the
    # runtime signature is unchanged.
    return cast("Callable[Concatenate[TicketRepository, P], Awaitable[R]]", wrapper)


def async_url(database_url: str) -> str:
    """Normalise a DSN to an async driver SQLAlchemy can use.

    psycopg 3 rather than asyncpg: it is already proven working here against the
    selector event loop, and adding a second driver would double the surface for
    driver-specific bugs.
    """
    if database_url.startswith("postgres"):
        if "+" not in database_url.split("://", 1)[0]:
            return database_url.replace("postgres://", "postgresql://", 1).replace(
                "postgresql://", "postgresql+psycopg://", 1
            )
        return database_url.replace("+asyncpg", "+psycopg")
    if database_url.startswith("sqlite") and "+" not in database_url.split("://", 1)[0]:
        return database_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return database_url


def _connect_args(url: str, settings: Settings) -> dict[str, Any]:
    """Driver-level deadlines, so a stall cannot outlive its usefulness.

    libpq keywords, and only valid for Postgres -- SQLite would reject them.
    `connect_timeout` bounds the handshake; the TCP keepalives make the kernel
    notice a peer that disappeared instead of retransmitting into the void; and
    `statement_timeout` stops a query that the server itself is sitting on.
    """
    if not url.startswith("postgresql"):
        return {}
    budget = max(1, int(settings.db_timeout_seconds))
    return {
        "connect_timeout": budget,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 5,
        "keepalives_count": 3,
        "options": f"-c statement_timeout={budget * 1000}",
    }


class TicketRepository:
    """Async persistence for tickets and CSAT responses."""

    backend = "postgres"

    # Consecutive failures before the store is presumed down, and how long to
    # leave it alone before trying again.
    BREAKER_THRESHOLD = 2
    BREAKER_COOLDOWN_SECONDS = 30.0

    def __init__(self, engine: AsyncEngine, timeout: float = 5.0) -> None:
        self._engine = engine
        self._session = async_sessionmaker(engine, expire_on_commit=False)
        self._timeout = timeout
        self._failures = 0
        self._opened_at: float | None = None

    # --- circuit breaker ---
    def _breaker_open(self) -> bool:
        """True while the store is presumed down and calls should skip it."""
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at < self.BREAKER_COOLDOWN_SECONDS:
            return True
        # Cooldown elapsed: let one call through to test the water. If it fails
        # the breaker re-opens; if it succeeds the store is back.
        self._opened_at = None
        return False

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.BREAKER_THRESHOLD and self._opened_at is None:
            self._opened_at = time.monotonic()
            log.warning(
                "repository.circuit_opened",
                extra={
                    "failures": self._failures,
                    "cooldown_seconds": self.BREAKER_COOLDOWN_SECONDS,
                },
            )

    def _record_success(self) -> None:
        if self._failures or self._opened_at is not None:
            log.info("repository.circuit_closed", extra={"after_failures": self._failures})
        self._failures = 0
        self._opened_at = None

    def circuit_state(self) -> str:
        """For the health report: `closed`, or `open` while shedding calls."""
        return "open" if self._breaker_open() else "closed"

    @classmethod
    async def connect(cls, settings: Settings | None = None) -> TicketRepository | None:
        """Open an engine, or return None so the caller keeps the in-memory store."""
        settings = settings or get_settings()
        if not settings.database_url:
            return None
        url = async_url(str(settings.database_url))
        try:
            engine = create_async_engine(
                url,
                pool_size=5,
                max_overflow=5,
                pool_pre_ping=True,
                # A connection idle longer than this is reopened rather than
                # trusted. Bounds how long a socket can outlive its peer.
                pool_recycle=300,
                connect_args=_connect_args(url, settings),
            )
            repo = cls(engine, timeout=settings.db_timeout_seconds)
            await repo._verify()
            log.info("repository.ready", extra={"backend": cls.backend})
            return repo
        except Exception as exc:
            log.warning("repository.unavailable", extra={"error": str(exc)[:250]})
            return None

    @_bounded
    async def _verify(self) -> None:
        """Fail fast if migrations have not been applied."""
        async with self._session() as session:
            await session.execute(select(Ticket).limit(1))

    async def close(self) -> None:
        await self._engine.dispose()

    # --- customers ---
    async def _ensure_customer(self, session: AsyncSession, customer_id: str) -> None:
        """Upsert a minimal customer row so the ticket foreign key holds."""
        existing = await session.get(Customer, customer_id)
        if existing is None:
            session.add(Customer(customer_id=customer_id, name="", email="", phone=""))

    @_bounded
    async def sync_customers(self, profiles: list[Any]) -> int:
        """Mirror the CRM's known customers into the database."""
        async with self._session() as session:
            for profile in profiles:
                row = await session.get(Customer, profile.customer_id)
                if row is None:
                    session.add(
                        Customer(
                            customer_id=profile.customer_id,
                            name=profile.name,
                            email=profile.email,
                            phone=profile.phone,
                            tier=profile.tier,
                        )
                    )
                else:
                    row.name, row.email, row.phone, row.tier = (
                        profile.name,
                        profile.email,
                        profile.phone,
                        profile.tier,
                    )
            await session.commit()
        return len(profiles)

    # --- tickets ---
    @_bounded
    async def save_ticket(self, ticket: Any) -> None:
        """Insert or update a ticket. Called on every ticketing-node run."""
        async with self._session() as session:
            await self._ensure_customer(session, ticket.customer_id)

            row = await session.get(Ticket, ticket.ticket_id)
            if row is None:
                row = Ticket(ticket_id=ticket.ticket_id, customer_id=ticket.customer_id)
                session.add(row)

            row.channel = ticket.channel
            row.subject = ticket.subject
            row.summary = ticket.summary
            row.status = ticket.status
            row.priority = ticket.priority
            row.intent = ticket.intent
            row.sentiment_score = ticket.sentiment_score
            row.escalated = ticket.escalated
            row.citations = ticket.citations
            row.handoff_packet = ticket.handoff_packet
            row.escalation_reason = getattr(ticket, "escalation_reason", None)
            row.assigned_to = getattr(ticket, "assigned_to", None)
            row.channels = list(getattr(ticket, "channels", None) or [])
            row.escalated_at = _to_dt(getattr(ticket, "escalated_at", None))
            row.first_responded_at = _to_dt(getattr(ticket, "first_responded_at", None))
            row.notified_at = _to_dt(getattr(ticket, "notified_at", None))
            row.conversation_id = getattr(ticket, "conversation_id", None) or (
                (ticket.handoff_packet or {}).get("conversation_id")
            )
            if ticket.handoff_packet and not row.escalation_reason:
                row.escalation_reason = ticket.handoff_packet.get("escalation_reason")

            if row.conversation_id:
                await self._save_transcript(
                    session,
                    conversation_id=row.conversation_id,
                    customer_id=ticket.customer_id,
                    channel=ticket.channel,
                    transcript=getattr(ticket, "transcript", []) or [],
                )

            await session.commit()

    async def _save_transcript(
        self,
        session: AsyncSession,
        *,
        conversation_id: str,
        customer_id: str,
        channel: str,
        transcript: list[dict[str, str]],
    ) -> None:
        """Replace the stored turns for a conversation.

        Replace rather than append: the ticketing node writes the whole
        transcript on every turn, so appending would duplicate every earlier
        turn once per turn.
        """
        if not transcript:
            return

        conversation = await session.get(Conversation, conversation_id)
        if conversation is None:
            conversation = Conversation(
                conversation_id=conversation_id,
                customer_id=customer_id,
                channels=[channel],
            )
            session.add(conversation)
        elif channel not in (conversation.channels or []):
            conversation.channels = [*(conversation.channels or []), channel]

        existing = await session.execute(
            select(Message).where(Message.conversation_id == conversation_id)
        )
        for stale in existing.scalars():
            await session.delete(stale)
        await session.flush()

        for row in transcript:
            session.add(
                Message(
                    conversation_id=conversation_id,
                    role=row.get("role", "customer"),
                    # Already PII-scrubbed by the ticketing node.
                    text=row.get("text", ""),
                    channel=channel,
                )
            )
        conversation.turn_count = len(transcript)

    async def _load_transcript(
        self, session: AsyncSession, conversation_id: str | None
    ) -> list[dict[str, str]]:
        if not conversation_id:
            return []
        result = await session.execute(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
        )
        return [{"role": m.role, "text": m.text} for m in result.scalars()]

    @_bounded
    async def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        async with self._session() as session:
            row = await session.get(Ticket, ticket_id)
            if row is None:
                return None
            record = _ticket_dict(row)
            record["transcript"] = await self._load_transcript(session, row.conversation_id)
            return record

    @_bounded
    async def list_tickets(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._session() as session:
            result = await session.execute(
                select(Ticket).order_by(Ticket.updated_at.desc()).limit(limit)
            )
            records = []
            for row in result.scalars():
                record = _ticket_dict(row)
                record["transcript"] = await self._load_transcript(session, row.conversation_id)
                records.append(record)
            return records

    @_bounded
    async def count_tickets(self) -> int:
        async with self._session() as session:
            result = await session.execute(select(Ticket))
            return len(list(result.scalars()))

    # --- feedback ---
    @_bounded
    async def save_feedback(
        self, *, ticket_id: str, score: int, comment: str, conversation_id: str | None
    ) -> None:
        async with self._session() as session:
            session.add(
                FeedbackResponse(
                    ticket_id=ticket_id,
                    conversation_id=conversation_id,
                    score=score,
                    comment=comment,
                )
            )
            await session.commit()

    @_bounded
    async def feedback_scores(self) -> list[int]:
        """Every submitted rating. CSAT is computed from these and nothing else."""
        async with self._session() as session:
            result = await session.execute(select(FeedbackResponse.score))
            return list(result.scalars())

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend, "durable": True}


def _to_dt(value: Any) -> Any:
    """Accept an epoch float or a datetime; store a timezone-aware datetime."""
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromtimestamp(float(value), tz=UTC)


def _epoch(value: datetime | None) -> float | None:
    """Back to an epoch float, which is what the CRM dataclass carries."""
    if value is None:
        return None
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return float(aware.timestamp())


def _ticket_dict(row: Ticket) -> dict[str, Any]:
    return {
        "ticket_id": row.ticket_id,
        "customer_id": row.customer_id,
        "conversation_id": row.conversation_id,
        "channel": row.channel,
        "subject": row.subject,
        "summary": row.summary,
        "status": row.status,
        "priority": row.priority,
        "intent": row.intent,
        "sentiment_score": row.sentiment_score,
        "escalated": row.escalated,
        "escalation_reason": row.escalation_reason,
        "escalated_at": _epoch(row.escalated_at),
        "first_responded_at": _epoch(row.first_responded_at),
        "notified_at": _epoch(row.notified_at),
        "assigned_to": row.assigned_to,
        "channels": row.channels or [],
        "citations": row.citations,
        "handoff_packet": row.handoff_packet,
        "created_at": row.created_at.timestamp() if row.created_at else 0.0,
        "updated_at": row.updated_at.timestamp() if row.updated_at else 0.0,
    }


_repository: TicketRepository | None = None


async def init_repository(settings: Settings | None = None) -> TicketRepository | None:
    global _repository
    _repository = await TicketRepository.connect(settings)
    return _repository


def get_repository() -> TicketRepository | None:
    """None when no database is configured; callers keep their in-memory store."""
    return _repository


async def shutdown_repository() -> None:
    global _repository
    if _repository is not None:
        await _repository.close()
        _repository = None
