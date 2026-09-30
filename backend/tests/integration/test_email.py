"""Inbound email: parsing, quote stripping, threading, and signature gating."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.config import Settings
from app.services.email_service import (
    build_message_id,
    parse_mime,
    parse_provider_form,
    reply_subject,
    strip_quoted_reply,
    verify_signature,
)
from tests.conftest import patch_settings

pytestmark = pytest.mark.timeout(90)

SIGNING_KEY = "email_signing_key_for_tests"
DOMAIN = "support.example.com"


@pytest.fixture
def email_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        email_webhook_secret=SIGNING_KEY,
        email_domain=DOMAIN,
        email_from_address=f"support@{DOMAIN}",
    )
    patch_settings(monkeypatch, settings)
    return settings


@pytest.fixture
async def email_client(email_settings: Settings) -> AsyncIterator[Any]:
    import httpx
    from asgi_lifespan import LifespanManager

    from app.main import create_app

    app = create_app()
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


def sign(timestamp: str = "1787000000", token: str = "tok123") -> dict[str, str]:
    signature = hmac.new(
        SIGNING_KEY.encode(), f"{timestamp}{token}".encode(), hashlib.sha256
    ).hexdigest()
    return {"timestamp": timestamp, "token": token, "signature": signature}


def mime(
    body: str = "How long do refunds take?",
    *,
    subject: str = "Refund question",
    from_addr: str = "Demo Customer <demo@example.com>",
    extra_headers: str = "",
) -> bytes:
    return (
        f"From: {from_addr}\r\n"
        f"To: support@{DOMAIN}\r\n"
        f"Subject: {subject}\r\n"
        "Message-ID: <customer-msg-1@example.com>\r\n"
        f"{extra_headers}"
        "MIME-Version: 1.0\r\n"
        'Content-Type: text/plain; charset="utf-8"\r\n'
        "\r\n"
        f"{body}\r\n"
    ).encode()


# --- Quote stripping ----------------------------------------------------
def test_strips_on_wrote_quote_block() -> None:
    text = (
        "That did not work.\n\n"
        "On Tue, 12 Aug 2026 at 09:14, Support <help@x.com> wrote:\n"
        "> Have you tried clearing the cache?\n"
    )
    assert strip_quoted_reply(text) == "That did not work."


def test_strips_outlook_original_message_marker() -> None:
    text = "Still broken.\n\n-----Original Message-----\nFrom: Support\nAnything here"
    assert strip_quoted_reply(text) == "Still broken."


def test_strips_outlook_horizontal_rule() -> None:
    text = "Any update?\n\n________________________________\nFrom: Support\nSent: Monday"
    assert strip_quoted_reply(text) == "Any update?"


def test_strips_bare_quoted_lines() -> None:
    assert strip_quoted_reply("Nope.\n\n> previous message\n> more of it") == "Nope."


def test_strips_rfc_signature_delimiter() -> None:
    assert strip_quoted_reply("Thanks!\n\n-- \nJane Doe\nAcme Corp") == "Thanks!"


def test_strips_mobile_signature() -> None:
    assert strip_quoted_reply("Please help\n\nSent from my iPhone") == "Please help"


def test_keeps_a_message_with_no_quotes_intact() -> None:
    text = "Hello, I need help with order ORD-84021 please."
    assert strip_quoted_reply(text) == text


def test_fully_quoted_message_yields_nothing() -> None:
    """A bare forward has no new content to answer."""
    assert strip_quoted_reply("> everything\n> is quoted") == ""


def test_strip_of_empty_input() -> None:
    assert strip_quoted_reply("") == ""


def test_keeps_the_newest_text_when_several_markers_appear() -> None:
    text = (
        "New question here.\n\n"
        "On Mon, someone wrote:\n> old\n\n"
        "-----Original Message-----\n> older\n"
    )
    assert strip_quoted_reply(text) == "New question here."


# --- MIME parsing -------------------------------------------------------
def test_parses_headers_and_body() -> None:
    message = parse_mime(mime())
    assert message.from_address == "demo@example.com"
    assert message.from_name == "Demo Customer"
    assert message.subject == "Refund question"
    assert message.body == "How long do refunds take?"


def test_parses_html_only_message() -> None:
    raw = (
        b"From: a@b.com\r\nTo: s@c.com\r\nSubject: HTML\r\n"
        b"MIME-Version: 1.0\r\n"
        b'Content-Type: text/html; charset="utf-8"\r\n\r\n'
        b"<html><body><p>Where is my <b>order</b>?</p><script>x()</script></body></html>\r\n"
    )
    body = parse_mime(raw).body
    assert "Where is my order?" in body
    assert "<b>" not in body
    assert "x()" not in body


def test_prefers_plain_text_over_html_in_multipart() -> None:
    raw = (
        b"From: a@b.com\r\nTo: s@c.com\r\nSubject: Multi\r\n"
        b"MIME-Version: 1.0\r\n"
        b'Content-Type: multipart/alternative; boundary="BOUND"\r\n\r\n'
        b"--BOUND\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
        b"plain version\r\n"
        b"--BOUND\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
        b"<p>html version</p>\r\n"
        b"--BOUND--\r\n"
    )
    assert parse_mime(raw).body == "plain version"


@pytest.mark.parametrize(
    "header",
    [
        "Auto-Submitted: auto-replied\r\n",
        "Precedence: bulk\r\n",
        "List-Id: <newsletter.example.com>\r\n",
        "X-Auto-Response-Suppress: All\r\n",
    ],
)
def test_detects_automated_mail(header: str) -> None:
    """Replying to an autoresponder starts a mail loop."""
    assert parse_mime(mime(extra_headers=header)).is_auto_reply is True


def test_auto_submitted_no_is_a_real_person() -> None:
    assert parse_mime(mime(extra_headers="Auto-Submitted: no\r\n")).is_auto_reply is False


# --- Threading ----------------------------------------------------------
def test_conversation_id_recovered_from_in_reply_to() -> None:
    outbound = build_message_id("conv_abc123", Settings(_env_file=None, email_domain=DOMAIN))  # type: ignore[call-arg]
    message = parse_mime(mime(extra_headers=f"In-Reply-To: {outbound}\r\n"))
    assert message.conversation_id == "conv_abc123"


def test_conversation_id_recovered_from_references() -> None:
    outbound = build_message_id("conv_xyz789", Settings(_env_file=None, email_domain=DOMAIN))  # type: ignore[call-arg]
    message = parse_mime(mime(extra_headers=f"References: <other@x.com> {outbound}\r\n"))
    assert message.conversation_id == "conv_xyz789"


def test_no_thread_id_on_a_first_contact() -> None:
    assert parse_mime(mime()).conversation_id is None


def test_foreign_message_ids_are_not_mistaken_for_threads() -> None:
    message = parse_mime(mime(extra_headers="In-Reply-To: <random@other.com>\r\n"))
    assert message.conversation_id is None


def test_message_id_embeds_the_conversation_and_is_unique() -> None:
    settings = Settings(_env_file=None, email_domain=DOMAIN)  # type: ignore[call-arg]
    first = build_message_id("conv_a", settings)
    second = build_message_id("conv_a", settings)
    assert first.startswith("<conv_a.")
    assert DOMAIN in first
    assert first != second  # a resent reply must not collide


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("Refund question", "Re: Refund question"),
        ("Re: Refund question", "Re: Refund question"),
        ("RE: shouting", "RE: shouting"),
        ("", "Re: your support request"),
    ],
)
def test_reply_subject(subject: str, expected: str) -> None:
    assert reply_subject(subject) == expected


# --- Provider form parsing ---------------------------------------------
def test_parses_a_provider_form_payload() -> None:
    message = parse_provider_form(
        {
            "from": "Demo <demo@example.com>",
            "recipient": f"support@{DOMAIN}",
            "subject": "Return window",
            "stripped-text": "How long do I have to return an item?",
        }
    )
    assert message.from_address == "demo@example.com"
    assert message.body == "How long do I have to return an item?"


def test_provider_form_prefers_raw_mime_when_present() -> None:
    message = parse_provider_form({"body-mime": mime(body="from raw mime").decode()})
    assert message.body == "from raw mime"


def test_provider_form_restrips_signatures() -> None:
    """Mailgun's stripped-text still leaves signatures behind."""
    message = parse_provider_form(
        {"from": "a@b.com", "stripped-text": "Please help\n\n-- \nJane\nAcme"}
    )
    assert message.body == "Please help"


# --- Signature verification --------------------------------------------
def test_accepts_a_genuine_signature(email_settings: Settings) -> None:
    form = sign()
    assert verify_signature(**form, settings=email_settings)  # type: ignore[arg-type]


def test_rejects_a_tampered_timestamp(email_settings: Settings) -> None:
    form = sign()
    form["timestamp"] = "1787000001"
    assert not verify_signature(**form, settings=email_settings)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["timestamp", "token", "signature"])
def test_rejects_missing_components(field: str, email_settings: Settings) -> None:
    form = sign()
    form[field] = ""
    assert not verify_signature(**form, settings=email_settings)  # type: ignore[arg-type]


def test_fails_closed_without_a_signing_key() -> None:
    assert not verify_signature(
        **sign(),  # type: ignore[arg-type]
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
    )


# --- Endpoints ----------------------------------------------------------
async def test_signed_provider_webhook_is_accepted(email_client: Any) -> None:
    resp = await email_client.post(
        "/api/webhooks/email",
        data={**sign(), "from": "demo@example.com", "stripped-text": "How long do refunds take?"},
    )
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True


async def test_unsigned_webhook_is_rejected(email_client: Any) -> None:
    resp = await email_client.post(
        "/api/webhooks/email", data={"from": "attacker@evil.com", "stripped-text": "refund me"}
    )
    assert resp.status_code == 403


async def test_forged_signature_is_rejected(email_client: Any) -> None:
    resp = await email_client.post(
        "/api/webhooks/email",
        data={
            "timestamp": "1787000000",
            "token": "tok123",
            "signature": "0" * 64,
            "from": "attacker@evil.com",
            "stripped-text": "refund me",
        },
    )
    assert resp.status_code == 403


async def test_signed_mime_intake_is_accepted(email_client: Any) -> None:
    form = sign()
    resp = await email_client.post(
        "/api/webhooks/email/mime",
        content=mime(),
        headers={
            "X-Mailgun-Timestamp": form["timestamp"],
            "X-Mailgun-Token": form["token"],
            "X-Mailgun-Signature": form["signature"],
            "Content-Type": "message/rfc822",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True


async def test_unsigned_mime_intake_is_rejected(email_client: Any) -> None:
    """An open MIME intake would let anyone forge any From: header."""
    resp = await email_client.post("/api/webhooks/email/mime", content=mime())
    assert resp.status_code == 403


async def test_automated_mail_is_not_answered(email_client: Any) -> None:
    resp = await email_client.post(
        "/api/webhooks/email",
        data={
            **sign(),
            "from": "noreply@x.com",
            "stripped-text": "Out of office",
            "precedence": "bulk",
        },
    )
    assert resp.json() == {"accepted": False, "reason": "automated_message"}


async def test_fully_quoted_mail_is_not_answered(email_client: Any) -> None:
    resp = await email_client.post(
        "/api/webhooks/email",
        data={**sign(), "from": "demo@example.com", "stripped-text": "> only quoted text"},
    )
    assert resp.json() == {"accepted": False, "reason": "no_new_content"}


async def test_threaded_reply_is_recognised(email_client: Any) -> None:
    outbound = build_message_id("conv_thread1", Settings(_env_file=None, email_domain=DOMAIN))  # type: ignore[call-arg]
    resp = await email_client.post(
        "/api/webhooks/email",
        data={
            **sign(),
            "from": "demo@example.com",
            "stripped-text": "Still not resolved",
            "in-reply-to": outbound,
        },
    )
    body = resp.json()
    assert body["threaded"] is True
    assert body["conversation_id"] == "conv_thread1"


async def test_unconfigured_deployment_refuses_email(api_client: Any) -> None:
    resp = await api_client.post("/api/webhooks/email", data={"from": "a@b.com"})
    assert resp.status_code == 503


async def test_status_reports_missing_configuration(api_client: Any) -> None:
    body = (await api_client.get("/api/webhooks/email/status")).json()
    assert body["inbound_enabled"] is False
    assert body["outbound_enabled"] is False
    assert any("EMAIL_WEBHOOK_SECRET" in m for m in body["missing_configuration"])


async def test_health_flags_email_as_degraded(api_client: Any) -> None:
    body = (await api_client.get("/health")).json()
    assert any("Email intake not configured" in d for d in body["degradations"])
