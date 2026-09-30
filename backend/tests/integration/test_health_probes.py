"""Health must report what is live, not what was wired at boot.

The failure this covers was observed, not imagined: with every container
stopped, `/health` still reported `crm.backend: postgres`,
`tickets_durable: true`, `sessions: redis` and `knowledge: qdrant`, and named
exactly one degradation -- an unrelated missing Twilio key. Every one of those
subsystems was unreachable at the time. `describe()` returns the backend the
process *chose* at startup, and nothing ever asked whether it still answered.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.api.routes.health import PROBE_TIMEOUT_SECONDS, _reachable

pytestmark = pytest.mark.timeout(60)


# --- One probe -------------------------------------------------------------
async def test_a_live_backend_is_reachable() -> None:
    async def ok() -> bool:
        return True

    assert await _reachable(ok) is True


async def test_a_raising_backend_is_unreachable() -> None:
    async def boom() -> bool:
        raise ConnectionRefusedError("connection refused")

    assert await _reachable(boom) is False


async def test_a_stalled_backend_is_unreachable_not_a_hang() -> None:
    """Slow past the budget and broken are the same answer to a caller."""

    async def forever() -> bool:
        await asyncio.Event().wait()
        return True

    async with asyncio.timeout(PROBE_TIMEOUT_SECONDS * 3):
        assert await _reachable(forever) is False


async def test_probe_budget_is_short_enough_to_be_useful() -> None:
    """A health check nobody waits for is not a health check."""
    assert PROBE_TIMEOUT_SECONDS <= 5


# --- The whole report ------------------------------------------------------
async def test_unreachable_stores_become_degradations(api_client: Any) -> None:
    """The regression: stores that stopped answering must show up as degraded."""
    body = (await api_client.get("/health")).json()

    assert "reachable" in body, "health does not probe liveness at all"
    for name, ok in body["reachable"].items():
        if ok is False:
            assert any(name in d for d in body["degradations"]), (
                f"{name} is unreachable but no degradation mentions it"
            )


async def test_a_dead_database_is_not_reported_as_durable(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`tickets_durable: true` against an unreachable Postgres is a lie."""

    class _Dead:
        async def count_tickets(self) -> int:
            raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr("app.services.repository.get_repository", lambda: _Dead())

    body = (await api_client.get("/health")).json()

    assert body["reachable"]["database"] is False
    assert body["subsystems"]["crm"]["reachable"] is False
    assert body["status"] == "degraded"
    assert any("database" in d for d in body["degradations"])


async def test_readiness_names_what_is_down(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Dead:
        async def count_tickets(self) -> int:
            raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr("app.services.repository.get_repository", lambda: _Dead())

    body = (await api_client.get("/health/ready")).json()
    assert body["status"] == "degraded"
    assert "database" in body["unreachable"]


async def test_readiness_still_serves_while_degraded(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Degraded is not unready: the in-process fallback can still answer."""

    class _Dead:
        async def count_tickets(self) -> int:
            raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr("app.services.repository.get_repository", lambda: _Dead())

    assert (await api_client.get("/health/ready")).status_code == 200


async def test_health_is_fast_even_when_everything_is_stalled(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probes run concurrently, so N dead stores cost one budget, not N."""

    class _Stalls:
        async def count_tickets(self) -> int:
            await asyncio.Event().wait()
            return 0

    monkeypatch.setattr("app.services.repository.get_repository", lambda: _Stalls())

    loop = asyncio.get_running_loop()
    started = loop.time()
    await api_client.get("/health")
    elapsed = loop.time() - started

    assert elapsed < PROBE_TIMEOUT_SECONDS * 3, f"/health took {elapsed:.1f}s"


async def test_unconfigured_subsystems_are_absent_not_false(api_client: Any) -> None:
    """Nothing to probe is a different claim from probed-and-dead."""
    body = (await api_client.get("/health")).json()
    # No database is configured in the offline test settings.
    assert "database" not in body["reachable"]
    assert body["subsystems"]["crm"]["reachable"] is None
