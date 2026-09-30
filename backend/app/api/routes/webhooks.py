"""Inbound webhook endpoints for SMS and WhatsApp.

Signature validation runs **before** the payload reaches any agent code, and
before the body is parsed into anything the graph sees. An unconfigured
deployment returns 503 rather than accepting unverifiable traffic.

Twilio expects a TwiML reply on the webhook response, so these endpoints answer
synchronously. The 15s Twilio timeout is the real constraint: if a turn takes
longer, the customer gets nothing. `_TURN_BUDGET_SECONDS` caps the agent run and
falls back to an acknowledgement rather than letting Twilio time out silently.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request, Response

from app.api.routes.chat import ChatRequest, run_turn
from app.config import get_settings
from app.services.crm import get_crm
from app.services.twilio_service import (
    SIGNATURE_HEADER,
    TwilioNotConfiguredError,
    empty_twiml,
    parse_inbound,
    twiml_response,
    verify_signature,
    webhook_url,
)
from app.utils.channel_adapter import render
from app.utils.logging import bind_context, get_logger, new_run_id
from app.utils.pii import mask

log = get_logger(__name__)

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])

# Twilio abandons a webhook at ~15s. Stay well inside it.
_TURN_BUDGET_SECONDS = 12.0

_TIMEOUT_REPLY = (
    "Thanks for your message. This one is taking longer than usual, so I have "
    "passed it to a colleague who will follow up shortly."
)
_FAILURE_REPLY = (
    "Sorry, something went wrong handling that message. Please try again, or "
    "reply HELP to reach a person."
)

# Carrier-level keywords Twilio already acts on. Replying would be duplicative
# and, for STOP, a compliance problem.
_OPT_OUT = frozenset({"stop", "stopall", "unsubscribe", "cancel", "end", "quit"})


async def _validated_form(request: Request, path: str) -> dict[str, str]:
    """Return the form body only if the Twilio signature checks out."""
    settings = get_settings()

    if not settings.twilio_webhooks_ready:
        # No token or no signed URL => validation is impossible => refuse.
        raise HTTPException(
            status_code=503,
            detail="Messaging webhooks are not configured on this deployment.",
        )

    raw = await request.form()
    params = {key: str(value) for key, value in raw.items()}

    try:
        url = webhook_url(settings, path, request.url.query)
    except TwilioNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not verify_signature(
        signature=request.headers.get(SIGNATURE_HEADER),
        url=url,
        params=params,
        settings=settings,
    ):
        # Do not echo the payload back; a forged request gets nothing useful.
        raise HTTPException(status_code=403, detail="Invalid signature.")

    return params


async def _handle(request: Request, path: str) -> Response:
    params = await _validated_form(request, path)
    message = parse_inbound(params)

    run_id = new_run_id()
    bind_context(channel=message.channel, langgraph_run_id=run_id)

    if message.body.lower() in _OPT_OUT:
        log.info("webhook.opt_out", extra={"channel": message.channel})
        return _xml(empty_twiml())

    if not message.body:
        # Media-only message: acknowledge without inventing a reply.
        log.info("webhook.no_text", extra={"num_media": message.num_media})
        return _xml(
            twiml_response(
                "Thanks — I can only read text messages at the moment. "
                "Could you describe the problem in a message?"
            )
        )

    # Resolve the caller from their number so the thread and tier are right.
    profile = await get_crm().identify(
        customer_id="", utterance=message.caller_id, channel=message.channel
    )
    customer_id = profile.customer_id if profile else f"anon:{message.caller_id}"

    log.info(
        "webhook.received",
        extra={
            "channel": message.channel,
            "identified": profile is not None,
            "pii_kinds": mask(message.body).kinds,
            "chars": len(message.body),
        },
    )

    payload = ChatRequest(
        message=message.body[:4000],
        customer_id=customer_id,
        channel=message.channel,
        customer_tier=profile.tier if profile else "standard",
    )

    try:
        turn = await asyncio.wait_for(run_turn(payload), timeout=_TURN_BUDGET_SECONDS)
    except TimeoutError:
        log.warning("webhook.turn_timeout", extra={"channel": message.channel})
        return _xml(twiml_response(_TIMEOUT_REPLY))
    except Exception:
        log.exception("webhook.turn_failed")
        return _xml(twiml_response(_FAILURE_REPLY))

    reply = render(turn["reply_text"], message.channel)
    return _xml(twiml_response(reply))


@router.post("/twilio/sms")
async def twilio_sms(request: Request) -> Response:
    """Inbound SMS. Replies as TwiML, segmented to 160 characters."""
    return await _handle(request, "/api/webhooks/twilio/sms")


@router.post("/twilio/whatsapp")
async def twilio_whatsapp(request: Request) -> Response:
    """Inbound WhatsApp. Replies as TwiML with WhatsApp markup."""
    return await _handle(request, "/api/webhooks/twilio/whatsapp")


@router.get("/status")
async def webhook_status() -> dict[str, object]:
    """What the messaging channels need, and whether they have it."""
    settings = get_settings()
    missing: list[str] = []
    if not settings.twilio_auth_token:
        missing.append("TWILIO_AUTH_TOKEN (required to verify signatures)")
    if not settings.twilio_webhook_base_url:
        missing.append("TWILIO_WEBHOOK_BASE_URL (the public URL Twilio calls)")
    if not settings.twilio_phone_number:
        missing.append("TWILIO_PHONE_NUMBER (required for outbound sends)")

    return {
        "inbound_enabled": settings.twilio_webhooks_ready,
        "missing_configuration": missing,
        "endpoints": {
            "sms": "/api/webhooks/twilio/sms",
            "whatsapp": "/api/webhooks/twilio/whatsapp",
        },
        "signature_validation": "always enforced; no bypass flag exists",
    }


def _xml(body: str) -> Response:
    return Response(content=body, media_type="application/xml")
