"""PII detection and reversible masking.

system specification Constraint: raw PII must never reach an LLM, a log line, or storage.
Text is scrubbed to synthetic tokens (`[[CC_1]]`) before inference; the mapping
stays server-side so outbound copy can be rehydrated if a channel needs it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

# Ordered most-specific first: SSN before generic digit runs, CC before phone.
_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b")),
    ("CC", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("EMAIL", re.compile(r"\b[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    (
        "PHONE",
        re.compile(
            r"(?<!\w)(?:\+\d{1,3}[ .-]?|\d{1,3}[ .-])?"
            r"(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)"
        ),
    ),
    # Real keys embed separators (sk_live_..., sk-proj-..., gsk_...), so the body
    # must admit _ and - or the match stops at the first separator.
    ("APIKEY", re.compile(r"\b(?:sk|pk|gsk|rk)[_-][A-Za-z0-9_-]{16,}\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
)

_SECRET_PHRASE = re.compile(
    r"\b(?:password|passwd|pwd|pin|secret|token|cvv|cvc)\b\s*(?:is|:|=)?\s*(\S+)",
    re.IGNORECASE,
)


def _luhn_valid(digits: str) -> bool:
    """Luhn checksum - keeps order numbers and IDs from being masked as cards."""
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for idx, digit in enumerate(reversed(nums)):
        if idx % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


@dataclass(slots=True)
class MaskResult:
    """Masked text plus the token -> original mapping."""

    text: str
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.mapping)

    @property
    def kinds(self) -> list[str]:
        """PII categories detected - safe to log (no values)."""
        return sorted({token.strip("[]").rsplit("_", 1)[0] for token in self.mapping})

    def unmask(self, text: str) -> str:
        """Rehydrate tokens. Only for outbound copy on a trusted channel."""
        for token, original in self.mapping.items():
            text = text.replace(token, original)
        return text


def mask(text: str) -> MaskResult:
    """Replace PII spans with stable synthetic tokens."""
    if not text:
        return MaskResult(text=text)

    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}
    # Reuse one token per distinct value so the LLM can still co-refer.
    seen: dict[tuple[str, str], str] = {}

    def _token_for(kind: str, value: str) -> str:
        key = (kind, value)
        if key in seen:
            return seen[key]
        counters[kind] = counters.get(kind, 0) + 1
        token = f"[[{kind}_{counters[kind]}]]"
        seen[key] = token
        mapping[token] = value
        return token

    def _sub(kind: str, pattern: re.Pattern[str], src: str) -> str:
        def repl(m: re.Match[str]) -> str:
            value = m.group(0)
            if kind == "CC" and not _luhn_valid(value):
                return value
            return _token_for(kind, value)

        return pattern.sub(repl, src)

    masked = text
    for kind, pattern in _PATTERNS:
        masked = _sub(kind, pattern, masked)

    # Credentials spoken in the clear ("my password is hunter2").
    def _secret_repl(m: re.Match[str]) -> str:
        return m.group(0).replace(m.group(1), _token_for("SECRET", m.group(1)))

    masked = _SECRET_PHRASE.sub(_secret_repl, masked)

    return MaskResult(text=masked, mapping=mapping)


def scrub(text: str) -> str:
    """Irreversibly redact - use for log lines and persisted transcripts."""
    return mask(text).text
