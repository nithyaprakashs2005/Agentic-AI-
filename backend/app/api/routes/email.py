"""Inbound email endpoints.

Two intake shapes, because providers differ:

* `POST /api/webhooks/email` -- a provider's parsed form payload, verified with
  the provider's webhook signature.
* `POST /api/webhooks/email/mime` -- raw RFC 822, for an MTA piping mail in over
  a private network. Also signature-gated, since an open MIME intake would let
  anyone impersonate any customer by forging `From:`.

Email is asynchronous: the customer is not waiting on the HTTP response. So
unlike the Twilio webhooks, the turn runs in the background and the endpoint
acknowledges immediately -- providers retry aggressively on a slow response, and
a retry would duplicate the turn.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.api.routes.chat import ChatRequest, run_turn
from app.config import get_settings
from app.services.crm import get_crm
from app.services.email_service import (
    InboundEmail,
    build_message_id,
    parse_mime,
    parse_provider_form,
    reply_subject,
    verify_signature,
)
from app.utils.channel_adapter import to_email
from app.utils.logging import bind_context, get_logger, new_run_id
from app.utils.pii import mask

log = get_logger(__name__)

router = APIRouter(prefix="/api/webhooks/email", tags=["email"])

MAX_BODY_CHARS = 8000


def _require_configured() -> Any:
    settings = get_settings()
    if not settings.email_webhooks_ready:
        raise HTTPException(
            status_code=503,
            detail="Email intake is not configured on this deployment.",
        )
    return settings


def _require_signature(form: dict[str, str]) -> None:
    """Verify the provider signature, or reject."""
    if not verify_signature(
        timestamp=form.get("timestamp", ""),
        token=form.get("token", ""),
        signature=form.get("signature", ""),
    ):
        raise HTTPException(status_code=403, detail="Invalid signature.")


async def _dispatch(message: InboundEmail) -> None:
    """Run the agent turn and record the reply. Executed in the background."""
    run_id = new_run_id()
    bind_context(channel="email", langgraph_run_id=run_id)

    profile = await get_crm().identify(
        customer_id="", utterance=message.from_address, channel="email"
    )
    customer_id = profile.customer_id if profile else f"anon:{message.from_address}"

    # The subject often carries the actual question ("Where is order ORD-1?"),
    # so include it when the body alone would be thin.
    text = message.body
    if message.subject and len(text) < 40:
        text = f"{message.subject}\n\n{text}".strip()

    payload = ChatRequest(
        message=text[:MAX_BODY_CHARS],
        customer_id=customer_id,
        channel="email",
        conversation_id=message.conversation_id,
        customer_tier=profile.tier if profile else "standard",
    )

    try:
        turn = await run_turn(payload)
    except Exception:
        log.exception("email.turn_failed", extra={"from_known": profile is not None})
        return

    reply_body = to_email(str(turn["reply_text"]))
    settings = get_settings()
    outbound_id = build_message_id(turn["conversation_id"], settings)

    # No outbound mail provider is wired, so this records the reply rather than
    # claiming a send. The Message-ID is what would let the customer's reply
    # rejoin this thread.
    log.info(
        "email.reply_prepared",
        extra={
            "conversation_id": turn["conversation_id"],
            "ticket_id": turn["state"].get("ticket_id"),
            "message_id": outbound_id,
            "in_reply_to": message.message_id,
            "subject": reply_subject(message.subject),
            "chars": len(reply_body),
            "delivered": False,
            "reason": "no outbound mail provider configured",
        },
    )


def _accept(message: InboundEmail, background: BackgroundTasks) -> dict[str, Any]:
    """Common intake decisions, after the signature has been verified."""
    if message.is_auto_reply:
        # Replying to an autoresponder starts a mail loop.
        log.info("email.auto_reply_ignored", extra={"from_domain": _domain(message)})
        return {"accepted": False, "reason": "automated_message"}

    if not message.body.strip():
        log.info("email.empty_after_stripping")
        return {"accepted": False, "reason": "no_new_content"}

    log.info(
        "email.received",
        extra={
            "from_domain": _domain(message),
            "threaded": message.conversation_id is not None,
            "pii_kinds": mask(message.body).kinds,
            "chars": len(message.body),
        },
    )
    background.add_task(_dispatch, message)
    return {
        "accepted": True,
        "threaded": message.conversation_id is not None,
        "conversation_id": message.conversation_id,
    }


@router.post("")
async def inbound_provider_webhook(request: Request, background: BackgroundTasks) -> dict[str, Any]:
    """Provider parsed-webhook intake."""
    _require_configured()
    raw = await request.form()
    form = {key: str(value) for key, value in raw.items()}
    _require_signature(form)
    return _accept(parse_provider_form(form), background)


@router.post("/mime")
async def inbound_mime(request: Request, background: BackgroundTasks) -> dict[str, Any]:
    """Raw RFC 822 intake, signature carried in headers."""
    _require_configured()
    body = await request.body()
    _require_signature(
        {
            "timestamp": request.headers.get("X-Mailgun-Timestamp", ""),
            "token": request.headers.get("X-Mailgun-Token", ""),
            "signature": request.headers.get("X-Mailgun-Signature", ""),
        }
    )
    return _accept(parse_mime(body), background)


@router.get("/status")
async def email_status() -> dict[str, Any]:
    """What the email channel needs, and whether it has it."""
    settings = get_settings()
    missing: list[str] = []
    if not settings.email_webhook_secret:
        missing.append("EMAIL_WEBHOOK_SECRET (required to verify provider signatures)")
    if not settings.email_domain:
        missing.append("EMAIL_DOMAIN (used in Message-IDs for thread continuity)")
    if not settings.email_from_address:
        missing.append("EMAIL_FROM_ADDRESS (required for outbound replies)")

    return {
        "inbound_enabled": settings.email_webhooks_ready,
        "outbound_enabled": False,
        "outbound_note": "no mail provider is wired; replies are logged, not sent",
        "missing_configuration": missing,
        "endpoints": {
            "provider_webhook": "/api/webhooks/email",
            "raw_mime": "/api/webhooks/email/mime",
        },
        "signature_validation": "always enforced; no bypass flag exists",
    }


def _domain(message: InboundEmail) -> str:
    """Sender domain -- safe to log, unlike the full address."""
    _, _, domain = message.from_address.partition("@")
    return domain or "unknown"
