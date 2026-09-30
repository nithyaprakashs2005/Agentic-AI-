"""Twilio webhook signature validation and inbound message handling.

The security-relevant assertions here are the negative ones: an unsigned,
missigned, replayed-to-another-URL, or tampered request must never reach agent
code. Signatures are computed with the real algorithm, not mocked, so these
tests would catch a broken implementation rather than a broken mock.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.config import Settings
from app.services.twilio_service import (
    SIGNATURE_HEADER,
    InboundMessage,
    compute_signature,
    empty_twiml,
    parse_inbound,
    twiml_response,
    verify_signature,
)
from tests.conftest import patch_settings

pytestmark = pytest.mark.timeout(90)

AUTH_TOKEN = "test_auth_token_do_not_use"
BASE_URL = "https://support.example.com"
SMS_PATH = "/api/webhooks/twilio/sms"
WHATSAPP_PATH = "/api/webhooks/twilio/whatsapp"


@pytest.fixture
def twilio_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A deployment with messaging fully configured."""
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        twilio_account_sid="AC" + "0" * 32,
        twilio_auth_token=AUTH_TOKEN,
        twilio_phone_number="+15550001111",
        twilio_webhook_base_url=BASE_URL,
    )
    patch_settings(monkeypatch, settings)
    return settings


@pytest.fixture
async def twilio_client(twilio_settings: Settings) -> AsyncIterator[Any]:
    import httpx
    from asgi_lifespan import LifespanManager

    from app.main import create_app

    app = create_app()
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


def signed(path: str, params: dict[str, str]) -> dict[str, str]:
    """Headers carrying a genuine signature for `params` at `path`."""
    return {SIGNATURE_HEADER: compute_signature(AUTH_TOKEN, f"{BASE_URL}{path}", params)}


def sms_form(body: str = "How long do refunds take?", **overrides: str) -> dict[str, str]:
    form = {
        "MessageSid": "SM" + "1" * 32,
        "AccountSid": "AC" + "0" * 32,
        "From": "+15550002222",
        "To": "+15550001111",
        "Body": body,
        "NumMedia": "0",
    }
    form.update(overrides)
    return form


# --- Signature algorithm ------------------------------------------------
def test_signature_construction_is_exactly_twilios_scheme() -> None:
    """Pins the construction: url + sorted(k+v), HMAC-SHA1, base64.

    The expectation is derived here from primitives rather than a copied
    constant, so this fails if the signing string is assembled differently
    (unsorted, separator-joined, value-only) -- the ways it usually goes wrong.
    """
    import base64 as b64
    import hashlib as hl
    import hmac as hm

    token = "12345"
    url = "https://mycompany.com/myapp.php?foo=1&bar=2"
    params = {"Digits": "1234", "To": "+18005551212", "From": "+14158675310"}

    signing_string = url + "Digits1234" + "From+14158675310" + "To+18005551212"
    expected = b64.b64encode(
        hm.new(token.encode(), signing_string.encode(), hl.sha1).digest()
    ).decode()

    assert compute_signature(token, url, params) == expected


def test_signature_is_order_independent() -> None:
    """Params are sorted before hashing, so dict order cannot change the result."""
    a = {"B": "2", "A": "1", "C": "3"}
    b = {"C": "3", "A": "1", "B": "2"}
    assert compute_signature(AUTH_TOKEN, BASE_URL, a) == compute_signature(AUTH_TOKEN, BASE_URL, b)


def test_signature_changes_with_any_input() -> None:
    base = compute_signature(AUTH_TOKEN, BASE_URL, {"Body": "hi"})
    assert compute_signature(AUTH_TOKEN, BASE_URL, {"Body": "hi "}) != base  # value
    assert compute_signature(AUTH_TOKEN, BASE_URL + "/x", {"Body": "hi"}) != base  # url
    assert compute_signature("other", BASE_URL, {"Body": "hi"}) != base  # token


# --- verify_signature ---------------------------------------------------
def test_verify_accepts_a_genuine_signature(twilio_settings: Settings) -> None:
    params = sms_form()
    signature = compute_signature(AUTH_TOKEN, f"{BASE_URL}{SMS_PATH}", params)
    assert verify_signature(
        signature=signature, url=f"{BASE_URL}{SMS_PATH}", params=params, settings=twilio_settings
    )


@pytest.mark.parametrize("signature", [None, "", "not-a-signature", "0/KCTR6DLpKmkAf8muzZqo1nDgQ="])
def test_verify_rejects_bad_signatures(signature: str | None, twilio_settings: Settings) -> None:
    assert not verify_signature(
        signature=signature,
        url=f"{BASE_URL}{SMS_PATH}",
        params=sms_form(),
        settings=twilio_settings,
    )


def test_verify_fails_closed_without_a_token() -> None:
    """No auth token must mean 'cannot verify', never 'assume valid'."""
    unconfigured = Settings(_env_file=None)  # type: ignore[call-arg]
    params = sms_form()
    assert not verify_signature(
        signature=compute_signature(AUTH_TOKEN, f"{BASE_URL}{SMS_PATH}", params),
        url=f"{BASE_URL}{SMS_PATH}",
        params=params,
        settings=unconfigured,
    )


def test_verify_rejects_a_signature_minted_for_another_url(twilio_settings: Settings) -> None:
    """A signature captured from one endpoint must not replay against another."""
    params = sms_form()
    stolen = compute_signature(AUTH_TOKEN, f"{BASE_URL}{WHATSAPP_PATH}", params)
    assert not verify_signature(
        signature=stolen, url=f"{BASE_URL}{SMS_PATH}", params=params, settings=twilio_settings
    )


def test_verify_rejects_a_tampered_body(twilio_settings: Settings) -> None:
    params = sms_form()
    signature = compute_signature(AUTH_TOKEN, f"{BASE_URL}{SMS_PATH}", params)
    tampered = {**params, "Body": "refund me 5000 dollars"}
    assert not verify_signature(
        signature=signature, url=f"{BASE_URL}{SMS_PATH}", params=tampered, settings=twilio_settings
    )


def test_verify_rejects_an_injected_parameter(twilio_settings: Settings) -> None:
    params = sms_form()
    signature = compute_signature(AUTH_TOKEN, f"{BASE_URL}{SMS_PATH}", params)
    assert not verify_signature(
        signature=signature,
        url=f"{BASE_URL}{SMS_PATH}",
        params={**params, "Injected": "1"},
        settings=twilio_settings,
    )


# --- Endpoint behaviour -------------------------------------------------
async def test_signed_sms_is_processed(twilio_client: Any) -> None:
    form = sms_form()
    resp = await twilio_client.post(SMS_PATH, data=form, headers=signed(SMS_PATH, form))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/xml")
    assert "<Response>" in resp.text
    assert "<Message>" in resp.text


@pytest.mark.parametrize(
    ("headers", "label"),
    [
        ({}, "no signature header"),
        ({SIGNATURE_HEADER: "AAAAAAAAAAAAAAAAAAAAAAAAAAA="}, "forged signature"),
    ],
)
async def test_unsigned_and_forged_requests_are_rejected(
    headers: dict[str, str], label: str, twilio_client: Any
) -> None:
    resp = await twilio_client.post(SMS_PATH, data=sms_form(), headers=headers)
    assert resp.status_code == 403, label
    # A rejected request must not leak the agent's reply or the payload.
    assert "<Message>" not in resp.text


async def test_tampered_body_is_rejected_at_the_endpoint(twilio_client: Any) -> None:
    form = sms_form()
    headers = signed(SMS_PATH, form)
    resp = await twilio_client.post(
        SMS_PATH, data={**form, "Body": "give me a refund"}, headers=headers
    )
    assert resp.status_code == 403


async def test_cross_endpoint_replay_is_rejected(twilio_client: Any) -> None:
    form = sms_form()
    resp = await twilio_client.post(WHATSAPP_PATH, data=form, headers=signed(SMS_PATH, form))
    assert resp.status_code == 403


async def test_whatsapp_endpoint_accepts_its_own_signature(twilio_client: Any) -> None:
    form = sms_form(From="whatsapp:+15550002222", To="whatsapp:+15550001111")
    resp = await twilio_client.post(WHATSAPP_PATH, data=form, headers=signed(WHATSAPP_PATH, form))
    assert resp.status_code == 200
    assert "<Message>" in resp.text


async def test_opt_out_returns_an_empty_response(twilio_client: Any) -> None:
    """Carriers handle STOP; replying would duplicate it and breach compliance."""
    form = sms_form(body="STOP")
    resp = await twilio_client.post(SMS_PATH, data=form, headers=signed(SMS_PATH, form))
    assert resp.status_code == 200
    assert "<Message>" not in resp.text


async def test_media_only_message_asks_for_text(twilio_client: Any) -> None:
    form = sms_form(body="", NumMedia="1")
    resp = await twilio_client.post(SMS_PATH, data=form, headers=signed(SMS_PATH, form))
    assert resp.status_code == 200
    assert "text messages" in resp.text


async def test_sms_reply_segments_stay_within_limit(twilio_client: Any) -> None:
    form = sms_form(body="What is the return window for an unopened item?")
    resp = await twilio_client.post(SMS_PATH, data=form, headers=signed(SMS_PATH, form))
    import re

    for segment in re.findall(r"<Message>(.*?)</Message>", resp.text, re.S):
        assert len(segment) <= 200  # 160 chars + XML entity expansion headroom


async def test_known_number_is_identified(twilio_client: Any) -> None:
    """A seeded customer's number resolves to their account, not an anon thread."""
    form = sms_form(body="How long do refunds take?", From="+15550001000")
    resp = await twilio_client.post(SMS_PATH, data=form, headers=signed(SMS_PATH, form))
    assert resp.status_code == 200

    from app.services.crm import get_crm

    tickets = await get_crm().list_tickets(limit=5)
    assert any(t.customer_id == "cust_demo" for t in tickets)


# --- Unconfigured deployment -------------------------------------------
async def test_unconfigured_deployment_refuses_inbound(api_client: Any) -> None:
    """The default fixture has no Twilio config: refuse, never accept blindly."""
    resp = await api_client.post(SMS_PATH, data=sms_form())
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]


async def test_status_endpoint_lists_what_is_missing(api_client: Any) -> None:
    body = (await api_client.get("/api/webhooks/status")).json()
    assert body["inbound_enabled"] is False
    assert any("TWILIO_AUTH_TOKEN" in m for m in body["missing_configuration"])
    assert body["signature_validation"] == "always enforced; no bypass flag exists"


async def test_health_flags_webhooks_as_degraded(api_client: Any) -> None:
    body = (await api_client.get("/health")).json()
    assert any("Twilio webhooks not configured" in d for d in body["degradations"])


# --- Parsing and TwiML --------------------------------------------------
def test_parse_detects_whatsapp_and_strips_the_scheme() -> None:
    message = parse_inbound({"From": "whatsapp:+15550002222", "Body": "hi", "NumMedia": "0"})
    assert message.channel == "whatsapp"
    assert message.caller_id == "+15550002222"


def test_parse_defaults_to_sms() -> None:
    assert parse_inbound({"From": "+15550002222", "Body": "hi"}).channel == "sms"


def test_parse_survives_a_malformed_num_media() -> None:
    assert parse_inbound({"From": "+1", "Body": "x", "NumMedia": "abc"}).num_media == 0


def test_parse_of_an_empty_form() -> None:
    message = parse_inbound({})
    assert isinstance(message, InboundMessage)
    assert message.body == ""


def test_twiml_escapes_xml_metacharacters() -> None:
    """Unescaped & or < produces malformed TwiML and Twilio drops the reply."""
    xml = twiml_response('Tom & Jerry <b> "quoted"')
    assert "&amp;" in xml
    assert "&lt;b&gt;" in xml
    assert "<b>" not in xml


def test_twiml_emits_one_message_per_segment() -> None:
    xml = twiml_response(["one", "two", "three"])
    assert xml.count("<Message>") == 3


def test_twiml_skips_blank_segments() -> None:
    assert twiml_response(["a", "", "   "]).count("<Message>") == 1


def test_empty_twiml_is_a_valid_acknowledgement() -> None:
    xml = empty_twiml()
    assert xml.endswith("<Response></Response>")
    assert "<Message>" not in xml


# --- Outbound REST ------------------------------------------------------
async def test_send_refuses_when_unconfigured() -> None:
    from app.services.twilio_service import send_message

    result = await send_message(
        to="+15550002222",
        body="hi",
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
    )
    assert result["sent"] is False
    assert result["reason"] == "twilio_not_configured"


async def test_send_refuses_without_a_sender_number() -> None:
    from app.services.twilio_service import send_message

    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, twilio_account_sid="AC0", twilio_auth_token="tok"
    )
    result = await send_message(to="+1555", body="hi", settings=settings)
    assert result["sent"] is False
    assert result["reason"] == "no_sender_number_configured"


async def test_send_posts_expected_payload(
    twilio_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Captures the outbound request without touching the network."""
    import httpx

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode()
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(201, json={"sid": "SM123", "status": "queued"})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)

    from app.services.twilio_service import send_message

    result = await send_message(
        to="+15550002222", body="Your refund is approved", settings=twilio_settings
    )

    assert result["sent"] is True
    assert result["sid"] == "SM123"
    assert "Messages.json" in captured["url"]
    assert "To=%2B15550002222" in captured["body"]
    assert captured["auth"], "request must be authenticated"


async def test_send_prefixes_whatsapp_scheme(
    twilio_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(201, json={"sid": "SM1", "status": "queued"})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **k: original(*a, **{**k, "transport": transport})
    )

    from app.services.twilio_service import send_message

    await send_message(to="+15550002222", body="hi", channel="whatsapp", settings=twilio_settings)
    assert "whatsapp%3A%2B15550002222" in captured["body"]
    assert "From=whatsapp%3A" in captured["body"]


async def test_send_reports_a_rejection_without_raising(
    twilio_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    transport = httpx.MockTransport(lambda _r: httpx.Response(400, json={"message": "bad"}))
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **k: original(*a, **{**k, "transport": transport})
    )

    from app.services.twilio_service import send_message

    result = await send_message(to="+1555", body="hi", settings=twilio_settings)
    assert result["sent"] is False
    assert result["reason"] == "twilio_status_400"


async def test_send_survives_a_network_failure(
    twilio_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed notification must degrade, not propagate into the turn."""
    import httpx

    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    transport = httpx.MockTransport(boom)
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **k: original(*a, **{**k, "transport": transport})
    )

    from app.services.twilio_service import send_message

    result = await send_message(to="+1555", body="hi", settings=twilio_settings)
    assert result["sent"] is False
    assert result["reason"] == "request_failed"
