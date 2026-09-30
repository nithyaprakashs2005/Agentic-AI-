"""Channel adaptation: voice plaintext, SMS segmentation, markup dialects."""

from __future__ import annotations

import pytest

from app.utils.channel_adapter import (
    SMS_SEGMENT_LIMIT,
    render,
    to_email,
    to_sms,
    to_voice,
    to_whatsapp,
)

MARKDOWN = """## Refund status

Your refund of **$247.50** was approved.

1. Open [your account](https://example.com/acct)
2. Email us at help@example.com

- Card refunds take 3-5 days
- Wallet refunds are instant

`REF-12345` is your reference.
"""


@pytest.mark.parametrize(
    "forbidden",
    ["**", "##", "```", "](", "http://", "https://", "- ", "$", "@", "%", "&"],
)
def test_voice_output_has_no_unspeakable_tokens(forbidden: str) -> None:
    """system specification Standard #10: no markdown, no URLs, no bare symbols."""
    spoken = to_voice(MARKDOWN)
    assert forbidden not in spoken


def test_voice_speaks_currency_as_words() -> None:
    spoken = to_voice("Your refund of $247.50 is approved.")
    assert "247 dollars and 50 cents" in spoken
    assert "$" not in spoken


def test_voice_speaks_whole_currency_without_cents() -> None:
    assert "89 dollars" in to_voice("The total is $89.00 today.")
    assert "1 dollar" in to_voice("A fee of $1 applies.")


def test_voice_replaces_urls_rather_than_reading_them() -> None:
    spoken = to_voice("Visit https://example.com/help for details.")
    assert "http" not in spoken
    assert "the link on your screen" in spoken


def test_voice_converts_at_symbol() -> None:
    assert " at " in to_voice("Email help@example.com")


def test_voice_ends_with_a_single_terminator() -> None:
    spoken = to_voice("First line\n\nSecond line")
    assert spoken.endswith(".")
    assert not spoken.endswith("..")


def test_voice_of_empty_input() -> None:
    assert to_voice("") == ""
    assert to_voice("   ") == ""


# --- SMS ---------------------------------------------------------------
def test_short_sms_is_one_unnumbered_segment() -> None:
    segments = to_sms("Your refund is approved.")
    assert segments == ["Your refund is approved."]


def test_long_sms_splits_and_numbers_segments() -> None:
    segments = to_sms("word " * 120)
    assert len(segments) > 1
    assert all(len(s) <= SMS_SEGMENT_LIMIT for s in segments)
    assert segments[0].endswith(f"(1/{len(segments)})")
    assert segments[-1].endswith(f"({len(segments)}/{len(segments)})")


def test_sms_never_splits_mid_word() -> None:
    segments = to_sms(" ".join(f"token{i}" for i in range(80)))
    rejoined = " ".join(s.rsplit(" (", 1)[0] for s in segments)
    assert "token1 " in rejoined + " "
    for segment in segments:
        assert not segment.startswith(" ")


def test_sms_respects_a_custom_limit() -> None:
    segments = to_sms("word " * 60, limit=70)
    assert all(len(s) <= 70 for s in segments)


def test_empty_sms_yields_no_segments() -> None:
    assert to_sms("") == []


# --- WhatsApp / email / webchat ----------------------------------------
def test_whatsapp_uses_single_asterisk_bold() -> None:
    out = to_whatsapp("Your refund of **$247.50** is approved.")
    assert "*$247.50*" in out
    assert "**" not in out


def test_whatsapp_keeps_links_as_text() -> None:
    out = to_whatsapp("See [your account](https://example.com/acct)")
    assert "https://example.com/acct" in out


def test_email_appends_a_signature() -> None:
    out = to_email("Your refund is approved.", agent_name="Support Team")
    assert out.endswith("--\nSupport Team")


def test_webchat_preserves_markdown() -> None:
    out = render(MARKDOWN, "webchat")
    assert "**$247.50**" in out
    assert "##" in out


# --- Dispatch ----------------------------------------------------------
def test_render_returns_list_only_for_sms() -> None:
    assert isinstance(render("hello", "sms"), list)
    for channel in ("voice", "webchat", "whatsapp", "email"):
        assert isinstance(render("hello", channel), str)  # type: ignore[arg-type]


def test_unknown_channel_falls_back_to_webchat() -> None:
    assert render("hello", "carrier-pigeon") == "hello"  # type: ignore[arg-type]
