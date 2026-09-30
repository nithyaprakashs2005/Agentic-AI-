"""A database that stalls must degrade quickly, not eventually.

Regression cover for a measured production-shaped failure: with Postgres
stopped *after* the process had connected, `/api/tickets/queue` returned 200
after 260 seconds. The in-memory fallback was correct and did run -- but only
once the OS gave up on a socket whose peer had vanished. Four minutes of
silence is indistinguishable from an outage to whoever is waiting.

The property under test is therefore about *time*, not just correctness: an
unreachable store has to surface as unavailable inside its budget.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from app.config import Settings
from app.services.crm import InMemoryCrm
from app.services.repository import (
    RepositoryTimeoutError,
    TicketRepository,
    _bounded,
    _connect_args,
    async_url,
)

pytestmark = pytest.mark.timeout(60)

BUDGET = 0.3


class _Budgeted(TicketRepository):
    """The smallest thing the deadline can attach to.

    Subclasses the real repository so the breaker bookkeeping is genuine, but
    skips `__init__` so no engine or driver is involved.
    """

    def __init__(self, timeout: float = BUDGET) -> None:
        self._engine = None  # type: ignore[assignment]
        self._timeout = timeout
        self._failures = 0
        self._opened_at = None


# --- The deadline itself ---------------------------------------------------
async def test_a_stalled_query_raises_inside_its_budget() -> None:
    """This is the whole fix: a hang becomes a prompt, catchable failure."""

    async def hangs_forever(self: Any) -> Any:
        await asyncio.Event().wait()

    bound = _bounded(hangs_forever)

    started = time.perf_counter()
    with pytest.raises(RepositoryTimeoutError):
        await bound(_Budgeted())
    elapsed = time.perf_counter() - started

    assert elapsed < BUDGET * 4, f"took {elapsed:.2f}s against a {BUDGET}s budget"


async def test_timeout_is_catchable_by_existing_callers() -> None:
    """Callers already guard with `except Exception`; do not slip past them."""
    assert issubclass(RepositoryTimeoutError, Exception)


async def test_a_healthy_call_is_untouched() -> None:
    """The deadline must add nothing to the path that works."""

    async def quick(self: Any, value: int) -> int:
        return value * 2

    assert await _bounded(quick)(_Budgeted(), 21) == 42


async def test_the_repository_takes_its_budget_from_settings() -> None:
    repo = TicketRepository(engine=None, timeout=2.5)  # type: ignore[arg-type]
    assert repo._timeout == 2.5


# --- The fallback the deadline exists to reach -----------------------------
async def test_crm_falls_back_to_memory_when_the_store_stalls(
    monkeypatch: pytest.MonkeyPatch, _offline_settings: Settings
) -> None:
    """The behaviour a user sees: the queue still answers, from memory."""
    crm = InMemoryCrm(_offline_settings)
    ticket = await crm.create_ticket(
        customer_id="cust_abc123",
        channel="webchat",
        subject="[billing] duplicate charge",
        intent="billing",
        priority="high",
        conversation_id="conv-stall",
    )

    class _Stalls:
        async def list_tickets(self, *, limit: int = 50) -> list[dict[str, Any]]:
            raise RepositoryTimeoutError("list_tickets")

        async def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
            raise RepositoryTimeoutError("get_ticket")

        async def save_ticket(self, ticket: Any) -> None:
            raise RepositoryTimeoutError("save_ticket")

    monkeypatch.setattr("app.services.repository.get_repository", lambda: _Stalls())

    started = time.perf_counter()
    rows = await crm.list_tickets(limit=10)
    elapsed = time.perf_counter() - started

    assert [t.ticket_id for t in rows] == [ticket.ticket_id]
    assert elapsed < 1.0


async def test_a_stalled_lookup_reports_absent_rather_than_failing(
    monkeypatch: pytest.MonkeyPatch, _offline_settings: Settings
) -> None:
    """An unreachable store must not turn a lookup into a 500."""

    class _Stalls:
        async def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
            raise RepositoryTimeoutError("get_ticket")

    monkeypatch.setattr("app.services.repository.get_repository", lambda: _Stalls())

    crm = InMemoryCrm(_offline_settings)
    assert await crm.get_ticket("TKT-not-in-memory") is None


async def test_a_stalled_write_does_not_fail_the_turn(
    monkeypatch: pytest.MonkeyPatch, _offline_settings: Settings
) -> None:
    """Losing durability on one ticket must not lose the customer's answer."""

    class _Stalls:
        async def save_ticket(self, ticket: Any) -> None:
            raise RepositoryTimeoutError("save_ticket")

    monkeypatch.setattr("app.services.repository.get_repository", lambda: _Stalls())

    crm = InMemoryCrm(_offline_settings)
    ticket = await crm.create_ticket(
        customer_id="cust_abc123",
        channel="webchat",
        subject="[general] hello",
        intent="general",
        priority="normal",
        conversation_id="conv-write",
    )
    updated = await crm.update_ticket(ticket.ticket_id, summary="still recorded")
    assert updated is not None
    assert updated.summary == "still recorded"


# --- Driver-level deadlines ------------------------------------------------
def test_postgres_gets_connect_and_statement_deadlines() -> None:
    settings = Settings(_env_file=None, db_timeout_seconds=4.0)  # type: ignore[call-arg]
    args = _connect_args("postgresql+psycopg://u:p@h/db", settings)

    assert args["connect_timeout"] == 4
    assert args["keepalives"] == 1
    assert "statement_timeout=4000" in args["options"]


def test_sqlite_is_left_alone() -> None:
    """libpq keywords would be rejected by any other driver."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert _connect_args(async_url("sqlite:///./x.db"), settings) == {}


def test_the_budget_is_configuration() -> None:
    assert Settings(_env_file=None, db_timeout_seconds=1.5).db_timeout_seconds == 1.5  # type: ignore[call-arg]


# --- Circuit breaker -------------------------------------------------------
class _Breakable(TicketRepository):
    """Real breaker logic, with the query replaced by a controllable stub."""

    BREAKER_COOLDOWN_SECONDS = 0.4

    def __init__(self, timeout: float = BUDGET) -> None:
        self._engine = None  # type: ignore[assignment]
        self._timeout = timeout
        self._failures = 0
        self._opened_at = None
        self.attempts = 0
        self.healthy = False

    @_bounded
    async def query(self) -> str:
        self.attempts += 1
        if self.healthy:
            return "ok"
        await asyncio.Event().wait()
        return "unreachable"


async def test_the_breaker_opens_after_repeated_failures() -> None:
    repo = _Breakable()

    for _ in range(TicketRepository.BREAKER_THRESHOLD):
        with pytest.raises(RepositoryTimeoutError):
            await repo.query()

    assert repo.circuit_state() == "open"


async def test_an_open_breaker_fails_instantly_instead_of_waiting() -> None:
    """The point of the breaker: stop paying the budget once the store is down."""
    repo = _Breakable()
    for _ in range(TicketRepository.BREAKER_THRESHOLD):
        with pytest.raises(RepositoryTimeoutError):
            await repo.query()

    attempts_before = repo.attempts
    started = time.perf_counter()
    with pytest.raises(RepositoryTimeoutError):
        await repo.query()
    elapsed = time.perf_counter() - started

    assert elapsed < BUDGET / 2, f"open circuit still waited {elapsed:.2f}s"
    assert repo.attempts == attempts_before, "open circuit still hit the database"


async def test_the_breaker_retries_after_its_cooldown() -> None:
    """A store that comes back must be picked up again without a restart."""
    repo = _Breakable()
    for _ in range(TicketRepository.BREAKER_THRESHOLD):
        with pytest.raises(RepositoryTimeoutError):
            await repo.query()
    assert repo.circuit_state() == "open"

    repo.healthy = True
    await asyncio.sleep(_Breakable.BREAKER_COOLDOWN_SECONDS + 0.1)

    assert await repo.query() == "ok"
    assert repo.circuit_state() == "closed"


async def test_a_success_resets_the_failure_count() -> None:
    """Occasional blips must not accumulate into an open circuit."""
    repo = _Breakable()
    with pytest.raises(RepositoryTimeoutError):
        await repo.query()

    repo.healthy = True
    assert await repo.query() == "ok"

    repo.healthy = False
    with pytest.raises(RepositoryTimeoutError):
        await repo.query()
    assert repo.circuit_state() == "closed", "one failure should not open the circuit"
