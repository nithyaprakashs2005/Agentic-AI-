"""LLM gateway: JSON extraction, rate shedding, fallback order, stub dispatch."""

from __future__ import annotations

from typing import Any

import pytest

from app.config import Settings
from app.services.llm import (
    LLMGateway,
    _is_rate_limit,
    _RateBucket,
    parse_json,
)


# --- JSON extraction ----------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here it is: {"a": 1}', {"a": 1}),
        ('{"a": 1} Hope that helps.', {"a": 1}),
        ("[1, 2, 3]", [1, 2, 3]),
        ('prose {"nested": {"b": 2}} more', {"nested": {"b": 2}}),
    ],
)
def test_parse_json_tolerates_model_noise(raw: str, expected: Any) -> None:
    assert parse_json(raw) == expected


@pytest.mark.parametrize("raw", ["", "not json at all", "{unclosed", None])
def test_parse_json_returns_the_default_on_junk(raw: Any) -> None:
    assert parse_json(raw or "", default={"fallback": True}) == {"fallback": True}


# --- Rate bucket --------------------------------------------------------
def test_bucket_allows_up_to_the_limit() -> None:
    bucket = _RateBucket(limit=3)
    for _ in range(3):
        assert bucket.allow()
        bucket.record()
    assert bucket.allow() is False


def test_penalise_saturates_the_window() -> None:
    """A real 429 must stop further attempts, not just count one."""
    bucket = _RateBucket(limit=5)
    bucket.record()
    bucket.penalise()
    assert bucket.allow() is False


def test_bucket_window_expires() -> None:
    bucket = _RateBucket(limit=1, window_seconds=0.0)
    bucket.record()
    assert bucket.allow() is True  # the window has already elapsed


@pytest.mark.parametrize(
    "message",
    [
        "RateLimitError: quota exceeded",
        "litellm: rate limit reached",
        "HTTP 429 Too Many Requests",
    ],
)
def test_rate_limit_detection(message: str) -> None:
    assert _is_rate_limit(RuntimeError(message)) is True


def test_non_rate_limit_errors_are_not_misread() -> None:
    assert _is_rate_limit(RuntimeError("connection refused")) is False


# --- Provider chain ordering -------------------------------------------
def test_chain_is_empty_when_nothing_is_configured() -> None:
    gateway = LLMGateway(Settings(_env_file=None))  # type: ignore[call-arg]
    gateway._ollama_available = False
    assert gateway._chain("slm") == []


def test_groq_precedes_gemini_precedes_ollama() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, groq_api_key="g", gemini_api_key="m"
    )
    gateway = LLMGateway(settings)
    assert [p.name for p in gateway._chain("frontier")] == [
        "groq",
        "gemini",
        "ollama",
    ]


def test_tier_selects_the_right_groq_model() -> None:
    settings = Settings(_env_file=None, groq_api_key="g")  # type: ignore[call-arg]
    gateway = LLMGateway(settings)
    slm = gateway._chain("slm")[0]
    frontier = gateway._chain("frontier")[0]
    assert slm.model == settings.groq_slm_model
    assert frontier.model == settings.groq_frontier_model
    assert slm.model != frontier.model


def test_ollama_is_dropped_once_probed_absent() -> None:
    gateway = LLMGateway(Settings(_env_file=None))  # type: ignore[call-arg]
    assert any(p.name == "ollama" for p in gateway._chain("slm"))
    gateway._ollama_available = False
    assert not any(p.name == "ollama" for p in gateway._chain("slm"))


# --- Stub behaviour -----------------------------------------------------
async def test_stub_is_flagged_and_never_claims_to_be_a_model() -> None:
    gateway = LLMGateway(Settings(_env_file=None))  # type: ignore[call-arg]
    gateway._ollama_available = False
    result = await gateway.complete([{"role": "user", "content": "hello"}], purpose="generic")
    assert result.stubbed is True
    assert result.provider == "stub"
    assert result.model == "offline-stub"


@pytest.mark.parametrize(
    ("text", "expected_intent"),
    [
        ("I was charged twice on my invoice", "billing"),
        ("my app keeps crashing on login", "technical"),
        ("I want to return this damaged item", "returns"),
        ("where is my package tracking", "orders"),
        ("random musings about nothing", "general"),
    ],
)
async def test_stub_intent_classification(text: str, expected_intent: str) -> None:
    gateway = LLMGateway(Settings(_env_file=None))  # type: ignore[call-arg]
    gateway._ollama_available = False
    result = await gateway.complete(
        [{"role": "user", "content": text}], purpose="intent", json_mode=True
    )
    assert result.json()["intent"] == expected_intent


async def test_stub_marks_disputes_as_complex() -> None:
    gateway = LLMGateway(Settings(_env_file=None))  # type: ignore[call-arg]
    gateway._ollama_available = False
    result = await gateway.complete(
        [{"role": "user", "content": "I want to dispute this fraud charge"}],
        purpose="intent",
        json_mode=True,
    )
    assert result.json()["resolution_path"] == "complex"


@pytest.mark.parametrize(
    ("text", "sign"),
    [
        ("this is absolutely terrible and unacceptable", -1),
        ("thanks so much, that was great", 1),
        ("what is my order status", 0),
    ],
)
async def test_stub_sentiment_direction(text: str, sign: int) -> None:
    gateway = LLMGateway(Settings(_env_file=None))  # type: ignore[call-arg]
    gateway._ollama_available = False
    result = await gateway.complete(
        [{"role": "user", "content": text}], purpose="sentiment", json_mode=True
    )
    score = result.json()["sentiment_score"]
    assert (score < 0) if sign < 0 else (score > 0) if sign > 0 else (score == 0)


async def test_stub_confidence_clears_the_threshold_for_a_clear_intent() -> None:
    """A plain FAQ must not escalate merely because we are offline."""
    gateway = LLMGateway(Settings(_env_file=None))  # type: ignore[call-arg]
    gateway._ollama_available = False
    result = await gateway.complete(
        [{"role": "user", "content": "how long does a refund take"}],
        purpose="intent",
        json_mode=True,
    )
    assert result.json()["confidence"] >= 0.70


async def test_status_reports_full_degradation() -> None:
    gateway = LLMGateway(Settings(_env_file=None))  # type: ignore[call-arg]
    gateway._ollama_available = False
    assert gateway.status()["degraded_to_stub"] is True


async def test_status_is_not_degraded_when_a_key_exists() -> None:
    gateway = LLMGateway(Settings(_env_file=None, groq_api_key="g"))  # type: ignore[call-arg]
    assert gateway.status()["degraded_to_stub"] is False


async def test_probe_ollama_caches_a_negative_result() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, ollama_base_url="http://127.0.0.1:1"
    )
    gateway = LLMGateway(settings)
    assert await gateway.probe_ollama() is False
    assert gateway._ollama_available is False
