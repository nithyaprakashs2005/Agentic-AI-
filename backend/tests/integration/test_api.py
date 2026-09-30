"""HTTP surface: chat, streaming, tickets, health, and error hygiene."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.timeout(90)


# --- Health -------------------------------------------------------------
async def test_liveness_needs_no_dependencies(api_client: Any) -> None:
    resp = await api_client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


async def test_readiness_reports_the_loaded_corpus(api_client: Any) -> None:
    body = (await api_client.get("/health/ready")).json()
    assert body["status"] == "ready"
    assert body["knowledge_documents"] > 0


async def test_health_declares_its_degradations_honestly(api_client: Any) -> None:
    """Offline, /health must say so rather than reporting a bare 'ok'."""
    body = (await api_client.get("/health")).json()
    assert body["status"] == "degraded"
    assert body["subsystems"]["llm"]["degraded_to_stub"] is True
    joined = " ".join(body["degradations"])
    assert "offline stub" in joined
    assert "Qdrant" in joined


async def test_health_never_leaks_secrets(api_client: Any) -> None:
    text = (await api_client.get("/health")).text
    for leak in ("api_key", "secret", "password", "SecretStr"):
        assert leak not in text.lower().replace("secret_key", "")


# --- Chat ---------------------------------------------------------------
async def test_chat_returns_a_grounded_answer(api_client: Any) -> None:
    resp = await api_client.post(
        "/api/chat",
        json={
            "message": "How long do refunds take to appear?",
            "customer_id": "cust_demo",
            "conversation_id": "api-1",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "billing"
    assert body["citations"]
    assert body["ticket_id"]
    assert body["stubbed"] is True  # honest offline labelling


async def test_chat_rejects_an_empty_message(api_client: Any) -> None:
    resp = await api_client.post("/api/chat", json={"message": ""})
    assert resp.status_code == 422


async def test_chat_rejects_an_oversized_message(api_client: Any) -> None:
    resp = await api_client.post("/api/chat", json={"message": "x" * 5000})
    assert resp.status_code == 422


async def test_chat_rejects_an_unknown_channel(api_client: Any) -> None:
    resp = await api_client.post("/api/chat", json={"message": "hi", "channel": "carrier-pigeon"})
    assert resp.status_code == 422


async def test_sms_channel_returns_segments(api_client: Any) -> None:
    resp = await api_client.post(
        "/api/chat",
        json={
            "message": "What is the return window for an unopened item?",
            "customer_id": "cust_demo",
            "channel": "sms",
            "conversation_id": "api-sms",
        },
    )
    body = resp.json()
    assert isinstance(body["reply"], list)
    assert all(len(segment) <= 160 for segment in body["reply"])


async def test_voice_channel_reply_is_speakable(api_client: Any) -> None:
    resp = await api_client.post(
        "/api/chat",
        json={
            "message": "How long do refunds take?",
            "customer_id": "cust_demo",
            "channel": "voice",
            "conversation_id": "api-voice",
        },
    )
    reply = resp.json()["reply"]
    assert isinstance(reply, str)
    for forbidden in ("**", "##", "http", "$", "@"):
        assert forbidden not in reply


async def test_conversation_id_is_honoured(api_client: Any) -> None:
    first = await api_client.post(
        "/api/chat",
        json={"message": "How long do refunds take?", "conversation_id": "api-thread"},
    )
    second = await api_client.post(
        "/api/chat",
        json={"message": "And bank transfers?", "conversation_id": "api-thread"},
    )
    assert first.json()["conversation_id"] == second.json()["conversation_id"] == "api-thread"


async def test_pii_in_a_message_does_not_appear_in_the_reply(api_client: Any) -> None:
    resp = await api_client.post(
        "/api/chat",
        json={
            "message": "My card 4111 1111 1111 1111 was charged twice",
            "customer_id": "cust_demo",
            "conversation_id": "api-pii",
        },
    )
    assert "4111 1111 1111 1111" not in resp.text


# --- Escalation and handoff --------------------------------------------
async def test_escalation_surfaces_awaiting_human(api_client: Any) -> None:
    resp = await api_client.post(
        "/api/chat",
        json={
            "message": "I want to speak to a human",
            "customer_id": "cust_demo",
            "conversation_id": "api-esc",
        },
    )
    body = resp.json()
    assert body["escalated"] is True
    assert body["awaiting_human"] is True
    assert body["escalation_reason"] == "explicit_keyword:human"


async def test_handoff_queue_exposes_the_packet(api_client: Any) -> None:
    await api_client.post(
        "/api/chat",
        json={
            "message": "get me a representative",
            "customer_id": "cust_demo",
            "conversation_id": "api-queue",
        },
    )
    body = (await api_client.get("/api/tickets/queue")).json()
    assert body["count"] >= 1
    entry = body["queue"][0]
    assert entry["status"] == "pending_human"
    assert entry["handoff_packet"]["executive_summary"]


async def test_resume_closes_the_handoff(api_client: Any) -> None:
    await api_client.post(
        "/api/chat",
        json={
            "message": "I need a human please",
            "customer_id": "cust_demo",
            "conversation_id": "api-resume",
        },
    )
    resp = await api_client.post(
        "/api/chat/resume",
        json={"conversation_id": "api-resume", "resolution": "handled by agent Dana"},
    )
    assert resp.status_code == 200
    assert resp.json()["awaiting_human"] is False


async def test_resume_on_an_unknown_conversation_is_404(api_client: Any) -> None:
    resp = await api_client.post("/api/chat/resume", json={"conversation_id": "does-not-exist"})
    assert resp.status_code == 404


async def test_resume_on_a_live_conversation_is_409(api_client: Any) -> None:
    await api_client.post(
        "/api/chat",
        json={"message": "How long do refunds take?", "conversation_id": "api-live"},
    )
    resp = await api_client.post("/api/chat/resume", json={"conversation_id": "api-live"})
    assert resp.status_code == 409


# --- Streaming ----------------------------------------------------------
async def test_sse_stream_emits_progress_then_the_message(api_client: Any) -> None:
    async with api_client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "message": "How long do refunds take?",
            "customer_id": "cust_demo",
            "conversation_id": "api-sse",
        },
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in resp.aiter_text()])

    for event in ("event: start", "event: node", "event: message", "event: done"):
        assert event in body, f"missing {event}"
    assert body.index("event: start") < body.index("event: message")


# --- Tickets ------------------------------------------------------------
async def test_ticket_lookup_and_404(api_client: Any) -> None:
    chat = await api_client.post(
        "/api/chat",
        json={"message": "How long do refunds take?", "conversation_id": "api-ticket"},
    )
    ticket_id = chat.json()["ticket_id"]

    found = await api_client.get(f"/api/tickets/{ticket_id}")
    assert found.status_code == 200
    assert found.json()["ticket_id"] == ticket_id
    assert found.json()["transcript"]

    missing = await api_client.get("/api/tickets/TKT-does-not-exist")
    assert missing.status_code == 404


async def test_ticket_transcript_is_pii_scrubbed(api_client: Any) -> None:
    chat = await api_client.post(
        "/api/chat",
        json={
            "message": "charge on card 4111 1111 1111 1111 is wrong",
            "customer_id": "cust_demo",
            "conversation_id": "api-scrub",
        },
    )
    ticket_id = chat.json()["ticket_id"]
    body = (await api_client.get(f"/api/tickets/{ticket_id}")).text
    assert "4111 1111 1111 1111" not in body
    assert "CC_1" in body  # masked token present instead


# --- Voice --------------------------------------------------------------
async def test_voice_status_reports_missing_configuration(api_client: Any) -> None:
    body = (await api_client.get("/api/voice/status")).json()
    assert body["enabled"] is False
    assert body["missing_configuration"]


async def test_voice_room_is_503_when_unconfigured(api_client: Any) -> None:
    resp = await api_client.post("/api/voice/rooms", json={"customer_id": "cust_demo"})
    assert resp.status_code == 503


async def test_voice_preview_flags_unspeakable_output(api_client: Any) -> None:
    body = (
        await api_client.post(
            "/api/voice/preview",
            json={"text": "Refund of **$247.50**. See https://example.com/x"},
        )
    ).json()
    assert body["checks"]["has_markdown"] is False
    assert body["checks"]["has_url"] is False
    assert body["checks"]["has_bare_symbols"] is False
    assert "247 dollars and 50 cents" in body["spoken"]


# --- Meta ---------------------------------------------------------------
async def test_root_and_openapi(api_client: Any) -> None:
    assert (await api_client.get("/")).status_code == 200
    schema = (await api_client.get("/openapi.json")).json()
    assert "/api/chat" in schema["paths"]


async def test_rate_limit_headers_are_present(api_client: Any) -> None:
    resp = await api_client.post(
        "/api/chat", json={"message": "hello", "conversation_id": "api-rl"}
    )
    assert "X-RateLimit-Limit" in resp.headers
    assert "X-RateLimit-Remaining" in resp.headers


async def test_health_is_exempt_from_rate_limiting(api_client: Any) -> None:
    """Health must stay reachable while the app is being hammered."""
    for _ in range(80):  # well past the 60/min default
        assert (await api_client.get("/health/live")).status_code == 200
