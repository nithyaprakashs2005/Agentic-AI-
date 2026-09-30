"""Sentiment evaluation and the escalation rule engine.

system specification Graph Nodes #6. Scoring is an SLM call; the escalation decision is
deterministic so it can be unit-tested at its boundaries and never varies with
model temperature.

Rules, in precedence order:
  1. explicit keyword ("agent", "human", "cancel my account", ...)  -> escalate
  2. sentiment below threshold AND enterprise tier                  -> escalate
  3. intent confidence below threshold                              -> escalate
  4. tool attempts over budget within one turn, still unresolved    -> escalate
  5. sustained negative trajectory across the whole window          -> escalate
"""

from __future__ import annotations

import re
import time
from typing import Any, Literal

from app.agents.prompts.system import SENTIMENT_PROMPT
from app.agents.state import (
    ESCALATION_KEYWORDS,
    SupportAgentState,
    is_smalltalk,
    last_user_text,
)
from app.config import Settings, get_settings
from app.services.llm import get_gateway
from app.utils.logging import get_logger
from app.utils.pii import mask

log = get_logger(__name__)

_REPEATED_COMPLAINT = re.compile(r"\b(?:third|3rd|fourth|4th|again|repeatedly)\b")


async def sentiment_eval_node(state: SupportAgentState) -> dict[str, Any]:
    started = time.perf_counter()
    settings = get_settings()
    question = last_user_text(state)

    score = 0.0
    provider = ""
    if question.strip():
        result = await get_gateway().complete(
            [
                {"role": "system", "content": SENTIMENT_PROMPT},
                {"role": "user", "content": mask(question).text},
            ],
            tier="slm",
            purpose="sentiment",
            temperature=0.0,
            max_tokens=120,
            json_mode=True,
        )
        parsed = result.json(default={}) or {}
        score = _clamp(parsed.get("sentiment_score", 0.0))
        provider = result.provider

    window = settings.sentiment_window_size
    trajectory = [*state.get("sentiment_trajectory", []), score][-window:]

    elapsed = round((time.perf_counter() - started) * 1000, 2)
    log.info(
        "sentiment.scored",
        extra={
            "sentiment_score": score,
            "trajectory": trajectory,
            "provider": provider,
            "latency_ms": elapsed,
        },
    )

    return {
        "sentiment_score": score,
        "sentiment_trajectory": trajectory,
        "node_timings_ms": {"sentiment": elapsed},
    }


def evaluate_escalation(
    state: SupportAgentState, settings: Settings | None = None
) -> tuple[bool, str | None]:
    """Deterministic escalation decision. Returns (escalate, reason)."""
    settings = settings or get_settings()

    # An upstream node may already have decided (e.g. failed identity check).
    if state.get("escalation_required"):
        return True, state.get("escalation_reason") or "upstream_request"

    text = last_user_text(state).lower()
    for keyword in ESCALATION_KEYWORDS:
        if keyword in text:
            return True, f"explicit_keyword:{keyword.replace(' ', '_')}"

    # Repeated fulfilment failures and a refund that has exceeded the published
    # outer bank-processing window are exceptions, not FAQ responses.
    if _REPEATED_COMPLAINT.search(text) and any(
        topic in text for topic in ("late", "delay", "refund", "wrong", "failed")
    ):
        return True, "repeated_complaint"
    if "refund" in text and ("10 days" in text or "ten days" in text) and any(
        phrase in text for phrase in ("still", "hasn't", "has not")
    ):
        return True, "refund_timing_exception"

    score = float(state.get("sentiment_score", 0.0))
    tier = state.get("customer_tier", "standard")
    if score < settings.sentiment_escalation_threshold and tier == "enterprise":
        return True, "negative_sentiment_enterprise"

    # Low confidence means "I could not understand the request" -- which is only
    # escalation-worthy when there *is* a request. A greeting is not ambiguous.
    confidence = float(state.get("intent_confidence", 1.0))
    if confidence < settings.confidence_escalation_threshold and not is_smalltalk(text):
        return True, "low_intent_confidence"

    attempts = int(state.get("tool_attempts", 0))
    if attempts > settings.max_tool_attempts and state.get("resolution_status") != "resolved":
        return True, "tool_attempts_exhausted"

    trajectory = [float(v) for v in state.get("sentiment_trajectory", [])]
    window = settings.sentiment_window_size
    if len(trajectory) >= window and all(
        value < settings.sentiment_escalation_threshold for value in trajectory[-window:]
    ):
        return True, "sustained_negative_trajectory"

    return False, None


def check_escalation(state: SupportAgentState) -> Literal["escalate", "continue"]:
    """Routing function for the conditional edge out of the sentiment node."""
    escalate, reason = evaluate_escalation(state)
    if escalate:
        log.info("escalation.triggered", extra={"reason": reason})
        return "escalate"
    return "continue"


def _clamp(value: Any, low: float = -1.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return 0.0
