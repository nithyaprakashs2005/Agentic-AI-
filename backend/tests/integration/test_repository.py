"""Durable ticket and feedback storage.

Runs against a temporary SQLite database so the suite stays hermetic; the same
code path serves PostgreSQL in deployment. The behaviour that matters is that a
ticket outlives the process that created it.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from app.services.repository import TicketRepository, async_url

pytestmark = pytest.mark.timeout(120)

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class _Ticket:
    """Minimal stand-in matching the CRM ticket shape the repository reads."""

    def __init__(self, ticket_id: str = "TKT-1", **overrides: Any) -> None:
        self.ticket_id = ticket_id
        self.customer_id = "cust_demo"
        self.channel = "webchat"
        self.subject = "Refund question"
        self.summary = "Customer asked about refund timing."
        self.status = "resolved"
        self.priority = "normal"
        self.intent = "billing"
        self.sentiment_score = 0.0
        self.escalated = False
        self.citations = [{"id": "kb-refund-timing", "relevance": 0.9}]
        self.handoff_packet: dict[str, Any] | None = None
        self.conversation_id: str | None = None
        self.transcript: list[dict[str, str]] = []
        for key, value in overrides.items():
            setattr(self, key, value)


@pytest.fixture
async def repo(tmp_path: Path) -> AsyncIterator[TicketRepository]:
    db = tmp_path / "repo_test.db"
    url = f"sqlite:///{db.as_posix()}"

    # Off-thread: a blocking subprocess would stall the test event loop.
    migrate = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_ROOT,
        env={**_env(), "DATABASE_URL": url, "PYTHONPATH": str(BACKEND_ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert migrate.returncode == 0, migrate.stderr

    from app.config import Settings

    repository = await TicketRepository.connect(
        Settings(_env_file=None, database_url=url)  # type: ignore[call-arg]
    )
    assert repository is not None, "repository should connect to a migrated database"
    yield repository
    await repository.close()


def _env() -> dict[str, str]:
    import os

    keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "COMSPEC", "PATHEXT")
    return {k: os.environ[k] for k in keep if k in os.environ}


# --- URL normalisation --------------------------------------------------
@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("postgres://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("postgresql+asyncpg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("sqlite:///./x.db", "sqlite+aiosqlite:///./x.db"),
        ("sqlite+aiosqlite:///./x.db", "sqlite+aiosqlite:///./x.db"),
    ],
)
def test_async_url_normalisation(given: str, expected: str) -> None:
    """A sync driver here would fail at runtime, not at import."""
    assert async_url(given) == expected


async def test_connect_returns_none_without_a_database() -> None:
    """No DATABASE_URL must leave the caller on its in-memory store, not crash."""
    from app.config import Settings

    assert await TicketRepository.connect(Settings(_env_file=None)) is None  # type: ignore[call-arg]


async def test_connect_returns_none_when_migrations_are_missing(tmp_path: Path) -> None:
    """An unmigrated database is a misconfiguration, not a usable store."""
    from app.config import Settings

    url = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
    assert (
        await TicketRepository.connect(
            Settings(_env_file=None, database_url=url)  # type: ignore[call-arg]
        )
        is None
    )


# --- Tickets ------------------------------------------------------------
async def test_ticket_round_trip(repo: TicketRepository) -> None:
    await repo.save_ticket(_Ticket())
    row = await repo.get_ticket("TKT-1")

    assert row is not None
    assert row["subject"] == "Refund question"
    assert row["citations"][0]["id"] == "kb-refund-timing"


async def test_save_is_an_upsert_not_a_duplicate(repo: TicketRepository) -> None:
    """The ticketing node writes on every turn; each must update, not insert."""
    await repo.save_ticket(_Ticket(status="open"))
    await repo.save_ticket(_Ticket(status="resolved"))

    assert await repo.count_tickets() == 1
    row = await repo.get_ticket("TKT-1")
    assert row is not None and row["status"] == "resolved"


async def test_handoff_packet_survives_as_structured_data(repo: TicketRepository) -> None:
    packet = {
        "escalation_reason": "explicit_keyword:human",
        "conversation_id": "conv_abc",
        "churn_risk": "high",
        "recommended_actions": ["Check the invoice history"],
    }
    await repo.save_ticket(_Ticket(escalated=True, status="pending_human", handoff_packet=packet))

    row = await repo.get_ticket("TKT-1")
    assert row is not None
    assert row["handoff_packet"]["churn_risk"] == "high"
    # Promoted to columns so the queue can filter without opening the JSON.
    assert row["escalation_reason"] == "explicit_keyword:human"
    assert row["conversation_id"] == "conv_abc"


async def test_unknown_customer_still_gets_a_ticket(repo: TicketRepository) -> None:
    """Inbound SMS and email produce customers we have never seen.

    The foreign key is kept for its integrity value, so an unknown contact is
    upserted rather than the constraint being dropped.
    """
    await repo.save_ticket(_Ticket("TKT-anon", customer_id="anon:+15550001111"))
    row = await repo.get_ticket("TKT-anon")
    assert row is not None and row["customer_id"] == "anon:+15550001111"


async def test_missing_ticket_is_none(repo: TicketRepository) -> None:
    assert await repo.get_ticket("TKT-nope") is None


async def test_list_is_newest_first(repo: TicketRepository) -> None:
    for i in range(3):
        await repo.save_ticket(_Ticket(f"TKT-{i}"))
    rows = await repo.list_tickets(limit=10)
    assert len(rows) == 3
    assert rows == sorted(rows, key=lambda r: -r["updated_at"])


async def test_list_respects_the_limit(repo: TicketRepository) -> None:
    for i in range(5):
        await repo.save_ticket(_Ticket(f"TKT-{i}"))
    assert len(await repo.list_tickets(limit=2)) == 2


# --- Customers ----------------------------------------------------------
async def test_customer_sync_is_idempotent(repo: TicketRepository) -> None:
    from app.services.crm import CustomerProfile

    profiles = [CustomerProfile("cust_a", "A", "a@x.com", "+1", "enterprise")]
    await repo.sync_customers(profiles)
    await repo.sync_customers(profiles)

    await repo.save_ticket(_Ticket("TKT-a", customer_id="cust_a"))
    row = await repo.get_ticket("TKT-a")
    assert row is not None


async def test_customer_sync_updates_a_changed_tier(repo: TicketRepository) -> None:
    """Tier drives escalation, so a stale copy would misroute conversations."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.models.tables import Customer
    from app.services.crm import CustomerProfile

    await repo.sync_customers([CustomerProfile("cust_b", "B", "b@x.com", "+1", "standard")])
    await repo.sync_customers([CustomerProfile("cust_b", "B", "b@x.com", "+1", "enterprise")])

    session_factory = async_sessionmaker(repo._engine)
    async with session_factory() as session:
        result = await session.execute(select(Customer).where(Customer.customer_id == "cust_b"))
        assert result.scalar_one().tier == "enterprise"


# --- Feedback -----------------------------------------------------------
async def test_feedback_round_trip(repo: TicketRepository) -> None:
    await repo.save_ticket(_Ticket())
    await repo.save_feedback(ticket_id="TKT-1", score=5, comment="great", conversation_id="c1")
    await repo.save_feedback(ticket_id="TKT-1", score=3, comment="", conversation_id="c1")

    assert sorted(await repo.feedback_scores()) == [3, 5]


async def test_no_feedback_yields_no_scores(repo: TicketRepository) -> None:
    """CSAT must stay null rather than being inferred from an empty set."""
    assert await repo.feedback_scores() == []


# --- Durability across processes ---------------------------------------
async def test_a_ticket_outlives_its_repository(tmp_path: Path) -> None:
    """The whole point: a restart must not lose tickets."""
    from app.config import Settings

    db = tmp_path / "durable.db"
    url = f"sqlite:///{db.as_posix()}"
    settings = Settings(_env_file=None, database_url=url)  # type: ignore[call-arg]

    # Off-thread: a blocking subprocess would stall the test event loop.
    migrate = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_ROOT,
        env={**_env(), "DATABASE_URL": url, "PYTHONPATH": str(BACKEND_ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert migrate.returncode == 0, migrate.stderr

    first = await TicketRepository.connect(settings)
    assert first is not None
    await first.save_ticket(_Ticket("TKT-survive", status="pending_human", escalated=True))
    await first.close()  # simulate the process ending

    second = await TicketRepository.connect(settings)
    assert second is not None
    try:
        row = await second.get_ticket("TKT-survive")
        assert row is not None, "ticket did not survive the connection lifecycle"
        assert row["status"] == "pending_human"
    finally:
        await second.close()


def test_describe_reports_durability(repo: TicketRepository) -> None:
    described = repo.describe()
    assert described["durable"] is True
    assert described["backend"] == "postgres"


# --- Transcripts --------------------------------------------------------
async def test_transcript_is_persisted_and_restored(repo: TicketRepository) -> None:
    """A ticket that outlives its process must bring its conversation with it.

    Tickets survived a restart before this, but the transcript came back empty:
    the `messages` table existed and nothing wrote to it.
    """
    ticket = _Ticket(
        "TKT-tr",
        conversation_id="conv_tr",
        transcript=[
            {"role": "customer", "text": "card [[CC_1]] charged twice"},
            {"role": "agent", "text": "Refunds take three to five days."},
        ],
    )
    await repo.save_ticket(ticket)

    row = await repo.get_ticket("TKT-tr")
    assert row is not None
    assert [t["role"] for t in row["transcript"]] == ["customer", "agent"]
    assert "[[CC_1]]" in row["transcript"][0]["text"]


async def test_repeated_writes_do_not_duplicate_turns(repo: TicketRepository) -> None:
    """The ticketing node writes the whole transcript every turn.

    Appending instead of replacing would duplicate every earlier turn once per
    turn, so a five-turn conversation would store fifteen messages.
    """
    ticket = _Ticket(
        "TKT-dup", conversation_id="conv_dup", transcript=[{"role": "customer", "text": "one"}]
    )
    await repo.save_ticket(ticket)

    ticket.transcript = [
        {"role": "customer", "text": "one"},
        {"role": "agent", "text": "two"},
    ]
    await repo.save_ticket(ticket)
    await repo.save_ticket(ticket)

    row = await repo.get_ticket("TKT-dup")
    assert row is not None
    assert len(row["transcript"]) == 2


async def test_ticket_without_a_conversation_has_an_empty_transcript(
    repo: TicketRepository,
) -> None:
    await repo.save_ticket(_Ticket("TKT-none"))
    row = await repo.get_ticket("TKT-none")
    assert row is not None and row["transcript"] == []


async def test_listing_carries_transcripts(repo: TicketRepository) -> None:
    await repo.save_ticket(
        _Ticket(
            "TKT-list",
            conversation_id="conv_list",
            transcript=[{"role": "customer", "text": "hello"}],
        )
    )
    rows = await repo.list_tickets(limit=5)
    assert any(r["transcript"] for r in rows)
