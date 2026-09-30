"""Inbound email parsing, thread continuity, and webhook verification.

system specification Phase 6 lists email as a first-class channel. Three problems have to
be solved for email specifically, none of which arise on chat or SMS:

1. **Quoted history.** A reply carries the entire prior thread. Feeding that to
   the model every turn wastes the context window and, worse, lets the agent
   answer a question the customer already had answered three replies ago.
   `strip_quoted_reply` cuts the message back to what was actually just written.

2. **Thread continuity.** Email has no session. Outbound replies carry a
   `Message-ID` that embeds the conversation id, so the customer's `In-Reply-To`
   hands the thread back to us on the next turn -- no fuzzy subject matching.

3. **Authenticity.** Anyone can forge a `From:` header. The provider's webhook
   signature is the only trustworthy signal, so it is verified before parsing,
   with the same fail-closed posture as the Twilio webhooks: unconfigured means
   refuse, never accept.

Mailgun's documented HMAC-SHA256 scheme is implemented concretely. Other
providers sign differently and need their own verifier -- see `verify_signature`.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import uuid
from dataclasses import dataclass, field
from email import message_from_bytes, policy
from email.utils import parseaddr

from app.config import Settings, get_settings
from app.utils.logging import get_logger

log = get_logger(__name__)

# Conversation id embedded in the local part: <conv_abc123.7f2a@domain>
_MESSAGE_ID_RE = re.compile(r"<\s*(conv_[a-z0-9]+)\.[a-z0-9]+@", re.IGNORECASE)

# --- Quoted-reply boundaries -------------------------------------------
# Ordered: the earliest match in the body wins, so the newest text is kept.
_QUOTE_MARKERS: tuple[re.Pattern[str], ...] = (
    # "On Tue, 12 Aug 2026 at 09:14, Support <x@y.com> wrote:"
    re.compile(r"^\s*On .{0,200}\bwrote:\s*$", re.MULTILINE),
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*Forwarded message\s*-{2,}\s*$", re.MULTILINE | re.IGNORECASE),
    # Outlook's horizontal rule, then From:/Sent:/To:
    re.compile(r"^\s*_{10,}\s*$", re.MULTILINE),
    re.compile(r"^\s*From:\s.+$\n^\s*Sent:\s", re.MULTILINE),
    # A run of quoted lines.
    re.compile(r"^\s*>.*(?:\n\s*>.*)*$", re.MULTILINE),
)

# Signature delimiters. RFC 3676 is "-- \n"; the rest are common in practice.
_SIGNATURE_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^--\s*$", re.MULTILINE),
    re.compile(r"^\s*Sent from my \w+.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Get Outlook for \w+.*$", re.MULTILINE | re.IGNORECASE),
)

# Bulk/automated mail we must never reply to: replying starts a mail loop.
_AUTO_HEADERS = ("auto-submitted", "x-auto-response-suppress", "list-id", "precedence")
_AUTO_PRECEDENCE = {"bulk", "junk", "list", "auto_reply"}


@dataclass(slots=True)
class InboundEmail:
    """Normalised inbound email."""

    message_id: str
    from_address: str
    from_name: str
    to_address: str
    subject: str
    body: str
    """Just the newly written text: quotes and signature removed."""
    in_reply_to: str = ""
    references: list[str] = field(default_factory=list)
    is_auto_reply: bool = False

    @property
    def conversation_id(self) -> str | None:
        """Thread id recovered from the headers we control, newest first."""
        for header in (self.in_reply_to, *reversed(self.references)):
            match = _MESSAGE_ID_RE.search(header or "")
            if match:
                return match.group(1)
        return None


def strip_quoted_reply(text: str) -> str:
    """Return only the text the sender just wrote.

    Cuts at the earliest quote boundary, then at a signature delimiter. A
    message that is *entirely* quoted (a bare forward) returns empty, which the
    caller treats as "nothing to answer".
    """
    if not text:
        return ""

    cut = len(text)
    for pattern in _QUOTE_MARKERS:
        match = pattern.search(text)
        if match and match.start() < cut:
            cut = match.start()
    body = text[:cut]

    for pattern in _SIGNATURE_MARKERS:
        match = pattern.search(body)
        if match:
            body = body[: match.start()]

    # Collapse the blank lines left behind by the cut.
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _best_body(message: object) -> str:
    """Prefer text/plain; fall back to text/html with tags removed."""
    from email.message import EmailMessage as StdEmailMessage

    assert isinstance(message, StdEmailMessage)

    plain = message.get_body(preferencelist=("plain",))
    if plain is not None:
        return str(plain.get_content())

    html = message.get_body(preferencelist=("html",))
    if html is not None:
        return _html_to_text(str(html.get_content()))
    return ""


def _html_to_text(html: str) -> str:
    """Crude HTML flattening -- enough for a support reply, not a renderer."""
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p\s*>", "\n\n", html)
    html = re.sub(r"(?s)<[^>]+>", "", html)
    for entity, char in (
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
    ):
        html = html.replace(entity, char)
    return re.sub(r"\n{3,}", "\n\n", html).strip()


def parse_mime(raw: bytes) -> InboundEmail:
    """Parse an RFC 822 message into a normalised inbound email."""
    message = message_from_bytes(raw, policy=policy.default)

    from_name, from_address = parseaddr(str(message.get("From", "")))
    _, to_address = parseaddr(str(message.get("To", "")))

    references = str(message.get("References", "")).split()
    body = strip_quoted_reply(_best_body(message))

    return InboundEmail(
        message_id=str(message.get("Message-ID", "")).strip(),
        from_address=from_address.lower(),
        from_name=from_name,
        to_address=to_address.lower(),
        subject=str(message.get("Subject", "")).strip(),
        body=body,
        in_reply_to=str(message.get("In-Reply-To", "")).strip(),
        references=references,
        is_auto_reply=_is_automated(message),
    )


def parse_provider_form(form: dict[str, str]) -> InboundEmail:
    """Parse a provider's parsed-webhook payload (Mailgun/SendGrid field names).

    Providers post pre-parsed fields rather than raw MIME. When the raw message
    is included, prefer it -- it is the authoritative form.
    """
    raw = form.get("body-mime") or form.get("email")
    if raw:
        return parse_mime(raw.encode("utf-8", errors="replace"))

    from_name, from_address = parseaddr(form.get("from") or form.get("From", ""))
    _, to_address = parseaddr(form.get("recipient") or form.get("to") or form.get("To", ""))

    text = (
        form.get("stripped-text")  # Mailgun already strips quotes
        or form.get("body-plain")
        or form.get("text")
        or _html_to_text(form.get("body-html") or form.get("html") or "")
    )
    # Re-strip even Mailgun's stripped-text: it leaves signatures behind.
    body = strip_quoted_reply(text)

    return InboundEmail(
        message_id=(form.get("Message-Id") or form.get("message-id") or "").strip(),
        from_address=from_address.lower(),
        from_name=from_name,
        to_address=to_address.lower(),
        subject=(form.get("subject") or form.get("Subject") or "").strip(),
        body=body,
        in_reply_to=(form.get("In-Reply-To") or form.get("in-reply-to") or "").strip(),
        references=(form.get("References") or form.get("references") or "").split(),
        is_auto_reply=_is_automated_form(form),
    )


def _is_automated(message: object) -> bool:
    from email.message import EmailMessage as StdEmailMessage

    assert isinstance(message, StdEmailMessage)
    for header in _AUTO_HEADERS:
        value = str(message.get(header, "")).strip().lower()
        if not value:
            continue
        if header == "precedence":
            if value in _AUTO_PRECEDENCE:
                return True
        elif header == "auto-submitted":
            if value != "no":
                return True
        else:
            return True
    return False


def _is_automated_form(form: dict[str, str]) -> bool:
    for header in _AUTO_HEADERS:
        value = (form.get(header) or "").strip().lower()
        if not value:
            continue
        if header == "precedence":
            if value in _AUTO_PRECEDENCE:
                return True
        elif header == "auto-submitted":
            if value != "no":
                return True
        else:
            return True
    return False


# --- Outbound identity --------------------------------------------------
def build_message_id(conversation_id: str, settings: Settings | None = None) -> str:
    """Message-ID embedding the conversation, so replies rejoin the thread."""
    settings = settings or get_settings()
    domain = settings.email_domain or "support.local"
    return f"<{conversation_id}.{uuid.uuid4().hex[:8]}@{domain}>"


def reply_subject(subject: str) -> str:
    """Prefix with Re: unless it already has one."""
    cleaned = subject.strip()
    if not cleaned:
        return "Re: your support request"
    return cleaned if cleaned.lower().startswith("re:") else f"Re: {cleaned}"


# --- Webhook verification ----------------------------------------------
def verify_signature(
    *,
    timestamp: str,
    token: str,
    signature: str,
    settings: Settings | None = None,
) -> bool:
    """Verify a Mailgun-style webhook signature.

    signature = hex(hmac_sha256(signing_key, timestamp + token))

    Returns False when unconfigured -- an unverifiable webhook is untrusted, and
    there is no bypass flag. Other providers (SendGrid ECDSA, Postmark basic
    auth) sign differently and need their own verifier here.
    """
    settings = settings or get_settings()

    if not settings.email_webhook_secret:
        log.error("email.validation_impossible", extra={"reason": "no signing key configured"})
        return False
    if not (timestamp and token and signature):
        log.warning("email.signature_incomplete")
        return False

    expected = hmac.new(
        settings.email_webhook_secret.get_secret_value().encode("utf-8"),
        f"{timestamp}{token}".encode(),
        hashlib.sha256,
    ).hexdigest()

    valid = hmac.compare_digest(expected, signature)
    if not valid:
        log.warning("email.signature_mismatch")
    return valid
