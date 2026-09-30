"""Twilio webhook signature validation and outbound messaging.

system specification Constraints, stated twice:
    NEVER bypass Twilio webhook signature validation on inbound endpoints.
    ALWAYS validate Twilio webhook signatures before processing inbound payloads.

There is deliberately **no bypass flag**. An unconfigured deployment rejects
inbound webhooks outright rather than offering a "skip validation in dev" escape
hatch, because that flag is exactly what gets left on in production.

Algorithm (Twilio's documented scheme for form-encoded webhooks):

    signing_string = full_url + concat(k + v for k, v in sorted(params))
    signature      = base64(hmac_sha1(auth_token, signing_string))

compared against the `X-Twilio-Signature` header in constant time.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.utils.logging import get_logger

log = get_logger(__name__)

SIGNATURE_HEADER = "X-Twilio-Signature"


class TwilioNotConfiguredError(RuntimeError):
    """No auth token available, so no inbound webhook can be trusted."""


def compute_signature(auth_token: str, url: str, params: dict[str, str]) -> str:
    """Twilio's request signature for a form-encoded webhook."""
    signing_string = url
    for key in sorted(params):
        signing_string += key + params[key]

    digest = hmac.new(
        auth_token.encode("utf-8"),
        signing_string.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def verify_signature(
    *,
    signature: str | None,
    url: str,
    params: dict[str, str],
    settings: Settings | None = None,
) -> bool:
    """Constant-time check of an inbound webhook signature.

    Returns False for a missing header or an unconfigured token -- never True by
    default. The caller turns False into a 403.
    """
    settings = settings or get_settings()

    if not settings.twilio_auth_token:
        # Cannot verify => cannot trust. Refuse rather than wave it through.
        log.error("twilio.validation_impossible", extra={"reason": "no auth token configured"})
        return False

    if not signature:
        log.warning("twilio.signature_missing")
        return False

    expected = compute_signature(settings.twilio_auth_token.get_secret_value(), url, params)
    valid = hmac.compare_digest(expected, signature)
    if not valid:
        # Never log the signatures themselves.
        log.warning("twilio.signature_mismatch", extra={"url": url, "param_count": len(params)})
    return valid


def webhook_url(settings: Settings, path: str, query: str = "") -> str:
    """Reconstruct the exact URL Twilio signed.

    Twilio signs the URL *it* called. Behind ngrok, a load balancer, or any TLS
    terminator, the URL the app sees differs from the public one (http vs https,
    internal host), and validation then fails for every legitimate request. So
    the public base is configuration, not inference.
    """
    base = (settings.twilio_webhook_base_url or "").rstrip("/")
    if not base:
        raise TwilioNotConfiguredError(
            "TWILIO_WEBHOOK_BASE_URL must be set to the public URL Twilio calls, "
            "otherwise signatures cannot be validated behind a proxy or tunnel."
        )
    url = f"{base}{path}"
    return f"{url}?{query}" if query else url


# --- Inbound payloads ---------------------------------------------------
@dataclass(slots=True)
class InboundMessage:
    """Normalised view of a Twilio SMS or WhatsApp webhook."""

    message_sid: str
    from_number: str
    to_number: str
    body: str
    channel: str  # "sms" | "whatsapp"
    num_media: int = 0

    @property
    def caller_id(self) -> str:
        """Bare E.164 number, with the WhatsApp scheme prefix stripped."""
        return self.from_number.removeprefix("whatsapp:")


def parse_inbound(form: dict[str, str]) -> InboundMessage:
    """Read Twilio's form fields into a normalised message."""
    from_number = form.get("From", "")
    channel = "whatsapp" if from_number.startswith("whatsapp:") else "sms"
    try:
        num_media = int(form.get("NumMedia", "0"))
    except ValueError:
        num_media = 0

    return InboundMessage(
        message_sid=form.get("MessageSid", ""),
        from_number=from_number,
        to_number=form.get("To", ""),
        body=form.get("Body", "").strip(),
        channel=channel,
        num_media=num_media,
    )


# --- Outbound TwiML -----------------------------------------------------
def _escape(text: str) -> str:
    """XML-escape message text.

    Customer copy can contain &, <, and >; unescaped they produce malformed
    TwiML and Twilio drops the whole response.
    """
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def twiml_response(messages: str | list[str]) -> str:
    """Build a TwiML <Response>.

    A list produces one <Message> per entry, which is how a multi-segment SMS
    reply from the channel adapter is delivered in order.
    """
    parts = [messages] if isinstance(messages, str) else list(messages)
    body = "".join(f"<Message>{_escape(part)}</Message>" for part in parts if part.strip())
    return f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>'


def empty_twiml() -> str:
    """Acknowledge without replying (media-only messages, opt-out keywords)."""
    return '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'


# --- Outbound REST (proactive sends) ------------------------------------
async def send_message(
    *, to: str, body: str, channel: str = "sms", settings: Settings | None = None
) -> dict[str, Any]:
    """Send a message via Twilio's REST API.

    Used for proactive notifications, not webhook replies (those go back as
    TwiML). Returns a result dict rather than raising, so a failed send degrades
    one notification instead of a whole turn.
    """
    settings = settings or get_settings()
    if not (settings.twilio_account_sid and settings.twilio_auth_token):
        return {"sent": False, "reason": "twilio_not_configured"}
    if not settings.twilio_phone_number:
        return {"sent": False, "reason": "no_sender_number_configured"}

    import httpx

    sender = settings.twilio_phone_number
    recipient = to
    if channel == "whatsapp":
        sender = f"whatsapp:{sender.removeprefix('whatsapp:')}"
        recipient = f"whatsapp:{to.removeprefix('whatsapp:')}"

    url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Messages.json"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                data={"To": recipient, "From": sender, "Body": body},
                auth=(
                    settings.twilio_account_sid,
                    settings.twilio_auth_token.get_secret_value(),
                ),
            )
    except Exception as exc:
        log.warning("twilio.send_failed", extra={"error": str(exc)[:200]})
        return {"sent": False, "reason": "request_failed"}

    if response.status_code >= 400:
        log.warning("twilio.send_rejected", extra={"status": response.status_code})
        return {"sent": False, "reason": f"twilio_status_{response.status_code}"}

    payload = response.json()
    log.info("twilio.sent", extra={"channel": channel, "sid": payload.get("sid")})
    return {"sent": True, "sid": payload.get("sid"), "status": payload.get("status")}
