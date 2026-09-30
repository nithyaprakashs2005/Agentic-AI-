"""Boundary-condition tests for every escalation rule.

system specification Standard #13: "All escalation rules must have boundary-condition
tests." Each rule is exercised just inside and just outside its threshold.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.agents.nodes.sentiment import check_escalation, evaluate_escalation
from app.agents.state import SupportAgentState, initial_state
from app.config import Settings


def make_state(
    message: str = "I need help with something specific", **overrides: object
) -> SupportAgentState:
    state = initial_state(customer_id="cust_demo", channel="webchat")
    state["messages"] = [HumanMessage(content=message)]
    # Default to values that do NOT escalate, so each test isolates one rule.
    state["intent"] = "billing"
    state["intent_confidence"] = 0.95
    state["sentiment_score"] = 0.0
    state["tool_attempts"] = 0
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# --- Rule 1: explicit keywords ------------------------------------------
@pytest.mark.parametrize(
    "message",
    [
        "let me speak to a human",
        "I want an agent",
        "get me a representative",
        "put me through to a manager",
        "I want to cancel my account",
        "I am contacting my lawyer",
        "can I talk to a real person",
        "delete my account please",
    ],
)
def test_explicit_keyword_escalates(message: str, _offline_settings: Settings) -> None:
    escalate, reason = evaluate_escalation(make_state(message), _offline_settings)
    assert escalate is True
    assert reason is not None and reason.startswith("explicit_keyword:")


@pytest.mark.parametrize(
    "message",
    [
        "the human resources page will not load",  # contains "human"
        "my manager account cannot sign in",  # contains "manager"
    ],
)
def test_keyword_substring_is_a_known_false_positive(
    message: str, _offline_settings: Settings
) -> None:
    """Documents current behaviour: matching is substring-based.

    Escalating these is the safe direction to fail -- a human sees a request that
    did not need one, rather than a customer being denied one they asked for.
    """
    escalate, reason = evaluate_escalation(make_state(message), _offline_settings)
    assert escalate is True
    assert reason is not None and reason.startswith("explicit_keyword:")


def test_no_keyword_does_not_escalate(_offline_settings: Settings) -> None:
    escalate, reason = evaluate_escalation(
        make_state("how long does a refund take"), _offline_settings
    )
    assert escalate is False
    assert reason is None


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("This is the third time my order is late.", "repeated_complaint"),
        ("My refund still hasn't arrived after 10 days.", "refund_timing_exception"),
    ],
)
def test_commerce_exceptions_escalate(
    message: str, reason: str, _offline_settings: Settings
) -> None:
    escalate, actual_reason = evaluate_escalation(make_state(message), _offline_settings)
    assert escalate is True
    assert actual_reason == reason


# --- Rule 2: negative sentiment + enterprise tier -----------------------
@pytest.mark.parametrize(
    ("score", "tier", "expected"),
    [
        (-0.31, "enterprise", True),  # just past the -0.3 threshold
        (-0.30, "enterprise", False),  # exactly at it: strict less-than
        (-0.29, "enterprise", False),
        (-0.99, "standard", False),  # rule is enterprise-only
        (-0.99, "free", False),
        (0.5, "enterprise", False),
    ],
)
def test_sentiment_tier_boundary(
    score: float, tier: str, expected: bool, _offline_settings: Settings
) -> None:
    state = make_state(sentiment_score=score, customer_tier=tier)
    escalate, reason = evaluate_escalation(state, _offline_settings)
    assert escalate is expected
    if expected:
        assert reason == "negative_sentiment_enterprise"


# --- Rule 3: low intent confidence --------------------------------------
@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.69, True),  # below the 0.70 threshold
        (0.70, False),  # exactly at it: strict less-than
        (0.71, False),
        (0.0, True),
        (1.0, False),
    ],
)
def test_confidence_boundary(
    confidence: float, expected: bool, _offline_settings: Settings
) -> None:
    state = make_state("my invoice looks wrong", intent_confidence=confidence)
    escalate, reason = evaluate_escalation(state, _offline_settings)
    assert escalate is expected
    if expected:
        assert reason == "low_intent_confidence"


@pytest.mark.parametrize("greeting", ["hello", "hi", "thanks", "ok", "no thanks", "bye"])
def test_smalltalk_is_exempt_from_confidence_rule(
    greeting: str, _offline_settings: Settings
) -> None:
    """A greeting is not an ambiguous request, so it must not fetch a human."""
    state = make_state(greeting, intent_confidence=0.1)
    escalate, _ = evaluate_escalation(state, _offline_settings)
    assert escalate is False


# --- Rule 4: tool attempts exhausted ------------------------------------
@pytest.mark.parametrize(
    ("attempts", "status", "expected"),
    [
        (3, "pending", True),  # over the budget of 2
        (2, "pending", False),  # exactly at it: strict greater-than
        (1, "pending", False),
        (99, "resolved", False),  # resolved overrides the counter
    ],
)
def test_tool_attempts_boundary(
    attempts: int, status: str, expected: bool, _offline_settings: Settings
) -> None:
    state = make_state(tool_attempts=attempts, resolution_status=status)
    escalate, reason = evaluate_escalation(state, _offline_settings)
    assert escalate is expected
    if expected:
        assert reason == "tool_attempts_exhausted"


# --- Rule 5: sustained negative trajectory ------------------------------
@pytest.mark.parametrize(
    ("trajectory", "expected"),
    [
        ([-0.4, -0.5, -0.6, -0.7, -0.8], True),  # full window, all below
        ([-0.4, -0.5, -0.6, -0.7], False),  # window not yet full (N=5)
        ([-0.4, -0.5, -0.2, -0.7, -0.8], False),  # one value above threshold
        ([-0.3, -0.4, -0.5, -0.6, -0.7], False),  # first value exactly at it
        ([0.1, 0.2, 0.3, 0.4, 0.5], False),
    ],
)
def test_trajectory_boundary(
    trajectory: list[float], expected: bool, _offline_settings: Settings
) -> None:
    state = make_state(sentiment_trajectory=trajectory)
    escalate, reason = evaluate_escalation(state, _offline_settings)
    assert escalate is expected
    if expected:
        assert reason == "sustained_negative_trajectory"


# --- Precedence and upstream override -----------------------------------
def test_upstream_escalation_is_respected(_offline_settings: Settings) -> None:
    state = make_state(escalation_required=True, escalation_reason="identity_verification_failed")
    escalate, reason = evaluate_escalation(state, _offline_settings)
    assert escalate is True
    assert reason == "identity_verification_failed"


def test_keyword_takes_precedence_over_sentiment(_offline_settings: Settings) -> None:
    """The most actionable reason should be reported, not merely the first true one."""
    state = make_state("I want a human", sentiment_score=-0.9, customer_tier="enterprise")
    _, reason = evaluate_escalation(state, _offline_settings)
    assert reason == "explicit_keyword:human"


def test_router_returns_string_labels(_offline_settings: Settings) -> None:
    assert check_escalation(make_state("I want a human")) == "escalate"
    assert check_escalation(make_state("how long does a refund take")) == "continue"


def test_thresholds_come_from_settings() -> None:
    """A tightened threshold must change the decision, proving no hardcoding."""
    strict = Settings(_env_file=None, confidence_escalation_threshold=0.99)  # type: ignore[call-arg]
    state = make_state("my invoice looks wrong", intent_confidence=0.95)
    escalate, reason = evaluate_escalation(state, strict)
    assert escalate is True
    assert reason == "low_intent_confidence"
