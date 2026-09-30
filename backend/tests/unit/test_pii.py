"""PII masking: detection, false positives, and reversibility."""

from __future__ import annotations

import pytest

from app.utils.pii import mask, scrub


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("my card is 4111 1111 1111 1111", "CC"),
        ("card 5500-0000-0000-0004 declined", "CC"),
        ("ssn 123-45-6789", "SSN"),
        ("email me at bob.smith@example.co.uk", "EMAIL"),
        ("call +1 555-123-4567", "PHONE"),
        ("call (555) 123-4567", "PHONE"),
        ("key sk_live_abcdef1234567890", "APIKEY"),
        ("server at 192.168.1.100", "IP"),
        ("iban GB33BUKB20201555555555", "IBAN"),
        ("my password is hunter2xyz", "SECRET"),
        ("the pin: 4821", "SECRET"),
    ],
)
def test_detects_pii(text: str, kind: str) -> None:
    result = mask(text)
    assert kind in result.kinds, f"expected {kind} in {result.kinds} for {text!r}"
    assert result.text != text


@pytest.mark.parametrize(
    "text",
    [
        "order 1234567890123 status",  # 13 digits, fails Luhn
        "reference 9876543210987654",  # 16 digits, fails Luhn
        "ORD-84021 has not arrived",
        "invoice INV-2026-0812 is wrong",
        "I was charged 247 dollars 50",
        "the 30 day return window",
        "version 2.0.1 crashed",
    ],
)
def test_no_false_positives_on_business_identifiers(text: str) -> None:
    """Order numbers and refs must survive: masking them breaks lookups."""
    assert scrub(text) == text


def test_luhn_gate_separates_cards_from_long_numbers() -> None:
    assert "CC" in mask("4111111111111111").kinds  # valid Luhn
    assert "CC" not in mask("4111111111111112").kinds  # invalid Luhn


def test_same_value_reuses_one_token() -> None:
    """Co-reference must survive masking so the model can still reason."""
    result = mask("charge bob@x.com then refund bob@x.com")
    assert result.text.count("[[EMAIL_1]]") == 2
    assert len(result.mapping) == 1


def test_distinct_values_get_distinct_tokens() -> None:
    result = mask("from bob@x.com to alice@y.com")
    assert "[[EMAIL_1]]" in result.text
    assert "[[EMAIL_2]]" in result.text
    assert len(result.mapping) == 2


def test_unmask_round_trips() -> None:
    original = "card 4111 1111 1111 1111 and mail bob@x.com"
    result = mask(original)
    assert result.unmask(result.text) == original


def test_empty_and_clean_text() -> None:
    assert mask("").text == ""
    assert mask("").found is False
    clean = mask("how long do refunds take")
    assert clean.found is False
    assert clean.kinds == []


def test_kinds_never_leak_values() -> None:
    """`kinds` is the log-safe projection: categories only, never contents."""
    result = mask("card 4111 1111 1111 1111, ssn 123-45-6789")
    assert result.kinds == ["CC", "SSN"]
    assert "4111" not in "".join(result.kinds)
