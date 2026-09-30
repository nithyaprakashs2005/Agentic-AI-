"""Per-channel output rendering.

system specification Standard #7: the LangGraph core is channel-agnostic. Every piece of
channel-specific formatting - voice plaintext, SMS segmentation, email
signatures, web markdown - lives here and nowhere else.
"""

from __future__ import annotations

import re
from typing import Final, Literal

Channel = Literal["voice", "sms", "whatsapp", "webchat", "email"]

SMS_SEGMENT_LIMIT: Final[int] = 160
WHATSAPP_LIMIT: Final[int] = 4096

# --- Markdown / structural stripping -------------------------------------
_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_ORDERED = re.compile(r"^\s*(\d+)[.)]\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_URL = re.compile(r"https?://\S+")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINEWLINE = re.compile(r"\n{3,}")

# Symbols a TTS engine renders badly or not at all.
_SPOKEN: Final[tuple[tuple[str, str], ...]] = (
    ("&", " and "),
    ("@", " at "),
    ("$", " dollars "),
    ("£", " pounds "),
    ("€", " euros "),
    ("%", " percent "),
    ("#", " number "),
    ("+", " plus "),
    ("=", " equals "),
    ("/", " slash "),
    ("~", " approximately "),
)

# "$247.50" -> "247 dollars and 50 cents" (run before generic symbol swap).
_CURRENCY = re.compile(r"\$\s?(\d[\d,]*)(?:\.(\d{2}))?")


def _strip_markdown(text: str) -> str:
    text = _CODE_FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _BOLD_ITALIC.sub(r"\2", text)
    text = _HEADING.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    return text


def _currency_to_words(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        whole = m.group(1).replace(",", "")
        cents = m.group(2)
        unit = "dollar" if whole == "1" else "dollars"
        if cents and cents != "00":
            return f"{whole} {unit} and {int(cents)} cents"
        return f"{whole} {unit}"

    return _CURRENCY.sub(repl, text)


def to_voice(text: str) -> str:
    """Flatten to speakable prose: no markdown, no URLs, no bare symbols.

    Lists become sentences joined with natural connectives so Kokoro does not
    read out bullet characters.
    """
    text = _strip_markdown(text)
    text = _URL.sub("the link on your screen", text)
    text = _currency_to_words(text)

    # Turn list structure into spoken transitions before dropping newlines.
    text = _ORDERED.sub(lambda m: "" if m.group(1) == "1" else "Next, ", text)
    text = _BULLET.sub("", text)

    for symbol, spoken in _SPOKEN:
        text = text.replace(symbol, spoken)

    text = text.replace("\n", ". ")
    text = re.sub(r"(?:\.\s*){2,}", ". ", text)
    text = _MULTISPACE.sub(" ", text)
    return text.strip(" .").strip() + "." if text.strip() else ""


def to_sms(text: str, *, limit: int = SMS_SEGMENT_LIMIT) -> list[str]:
    """Plain text split into segments on sentence/word boundaries.

    Returns a list so the caller sends N messages rather than truncating.
    Segments are suffixed "(1/3)" when there is more than one.
    """
    flat = _strip_markdown(text)
    flat = _BULLET.sub("- ", flat)
    flat = _MULTINEWLINE.sub("\n", flat).strip()
    flat = _MULTISPACE.sub(" ", flat)

    if len(flat) <= limit:
        return [flat] if flat else []

    # Reserve room for the " (n/m)" counter.
    budget = limit - 8
    words = flat.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > budget and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)

    total = len(chunks)
    return [f"{c} ({i}/{total})" for i, c in enumerate(chunks, start=1)]


def to_whatsapp(text: str) -> str:
    """WhatsApp supports light markup: *bold*, _italic_, and long messages."""
    text = _CODE_FENCE.sub(" ", text)
    text = _MD_LINK.sub(r"\1: \2", text)
    text = _HEADING.sub("", text)
    # Markdown **bold** -> WhatsApp *bold*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = _MULTINEWLINE.sub("\n\n", text).strip()
    return text[:WHATSAPP_LIMIT]


def to_webchat(text: str) -> str:
    """Web chat renders GitHub-flavoured markdown; pass through, normalised."""
    return _MULTINEWLINE.sub("\n\n", text).strip()


def to_email(text: str, *, agent_name: str = "Support Team") -> str:
    body = _MULTINEWLINE.sub("\n\n", text).strip()
    return f"{body}\n\n--\n{agent_name}"


def render(
    text: str,
    channel: str,
    *,
    limit: int = SMS_SEGMENT_LIMIT,
    agent_name: str = "Support Team",
) -> str | list[str]:
    """Dispatch to the renderer for `channel`.

    SMS returns list[str] (one entry per segment); every other channel a str.

    `channel` is typed `str`, not `Channel`, because the value arrives off the
    wire: an unrecognised channel must degrade to web-chat rendering rather than
    raise. Callers holding a validated `Channel` can pass it directly.
    """
    match channel:
        case "voice":
            return to_voice(text)
        case "sms":
            return to_sms(text, limit=limit)
        case "whatsapp":
            return to_whatsapp(text)
        case "email":
            return to_email(text, agent_name=agent_name)
        case _:
            # Covers "webchat" and anything unrecognised.
            return to_webchat(text)
