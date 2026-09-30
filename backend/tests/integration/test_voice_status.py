"""Voice status must describe what is running, not what was configured.

Same defect class as the health endpoint: `enabled` was computed purely from
`LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` being present. With both set and
nothing listening on the port, this reported `enabled: true` -- so a console
could reasonably tell someone the voice pipeline was up while no call could be
placed. Credentials are a precondition, not evidence.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.api.routes.voice import LIVEKIT_PROBE_SECONDS, _livekit_reachable

pytestmark = pytest.mark.timeout(60)


# --- The probe -------------------------------------------------------------
async def test_a_closed_port_is_not_reachable() -> None:
    # Port 1 is reserved and never bound.
    assert await _livekit_reachable("ws://127.0.0.1:1") is False


async def test_a_listening_port_is_reachable() -> None:
    server = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        assert await _livekit_reachable(f"ws://127.0.0.1:{port}") is True
    finally:
        server.close()
        await server.wait_closed()


async def test_an_unresolvable_host_is_not_reachable() -> None:
    assert await _livekit_reachable("ws://livekit.invalid:7880") is False


async def test_a_malformed_url_does_not_raise() -> None:
    """Status must never 500 because a URL was typed badly."""
    assert await _livekit_reachable("not a url") is False


async def test_the_probe_is_bounded() -> None:
    """Nobody waits on a status endpoint."""
    assert LIVEKIT_PROBE_SECONDS <= 3


# --- The reported status ---------------------------------------------------
async def test_status_is_not_enabled_without_a_live_server(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression: keys present, nothing listening, so not enabled."""
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "livekit_api_key", _secret("devkey"), raising=False)
    monkeypatch.setattr(settings, "livekit_api_secret", _secret("devsecret"), raising=False)
    monkeypatch.setattr(settings, "livekit_url", "ws://127.0.0.1:1", raising=False)

    body = (await api_client.get("/api/voice/status")).json()

    assert body["credentials_configured"] is True
    assert body["transport"]["reachable"] is False
    assert body["enabled"] is False, "reported enabled with no server listening"
    assert any("listening" in m for m in body["missing_configuration"])


async def test_status_distinguishes_unconfigured_from_unreachable(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No keys is a different problem from a server that needs starting."""
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "livekit_api_key", None, raising=False)
    monkeypatch.setattr(settings, "livekit_api_secret", None, raising=False)

    body = (await api_client.get("/api/voice/status")).json()

    assert body["credentials_configured"] is False
    assert body["enabled"] is False
    assert any("LIVEKIT_API_KEY" in m for m in body["missing_configuration"])


async def test_status_reports_whether_the_tts_model_exists(api_client: Any) -> None:
    """Kokoro is a downloaded artefact, so its presence is checkable."""
    body = (await api_client.get("/api/voice/status")).json()
    assert isinstance(body["tts"]["model_present"], bool)


async def test_status_always_answers(api_client: Any) -> None:
    response = await api_client.get("/api/voice/status")
    assert response.status_code == 200
    assert "target_ttfa_ms" in response.json()


# --- Voice rendering, which works with no voice infrastructure at all ------
async def test_preview_strips_what_reads_badly_aloud(api_client: Any) -> None:
    response = await api_client.post(
        "/api/voice/preview",
        json={"text": "**Refunds** take 3-5 days. See https://kb.example.com/refunds"},
    )
    body = response.json()

    assert body["checks"]["has_markdown"] is False
    assert body["checks"]["has_url"] is False
    assert "**" not in body["spoken"]


def _secret(value: str) -> Any:
    from pydantic import SecretStr

    return SecretStr(value)
