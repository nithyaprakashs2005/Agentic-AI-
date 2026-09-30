"""Router node: intent classification and resolution-path selection.

SLM tier, target <30ms on Groq. This node is the *sole owner* of
`resolution_path` -- the conditional edge downstream only reads the value it
sets (system specification Graph Nodes #1).
"""

from __future__ import annotations

import time
from typing import Any

from app.agents.prompts.system import ROUTER_PROMPT
from app.agents.state import INTENTS, SupportAgentState, last_user_text
from app.services.llm import get_gateway
from app.utils.logging import get_logger
from app.utils.pii import mask

log = get_logger(__name__)


async def router_node(state: SupportAgentState) -> dict[str, Any]:
    started = time.perf_counter()
    question = last_user_text(state)

    if not question.strip():
        return {
            "intent": "general",
            "intent_confidence": 0.0,
            "resolution_path": "simple",
            "model_tier": "slm",
            "node_timings_ms": {"router": 0.0},
        }

    # PII never reaches the model (system specification Constraints).
    masked = mask(question)

    result = await get_gateway().complete(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": masked.text},
        ],
        tier="slm",
        purpose="intent",
        temperature=0.0,
        max_tokens=200,
        json_mode=True,
    )

    parsed = result.json(default={}) or {}

    intent = str(parsed.get("intent", "general")).lower().strip()
    if intent not in INTENTS:
        # An unknown label is a classification failure, not a category. Treat it
        # as low-confidence general so the escalation rules can catch it.
        log.warning("router.unknown_intent", extra={"raw_intent": intent[:40]})
        intent = "general"
        confidence = 0.0
    else:
        confidence = _clamp(parsed.get("confidence", 0.0))

    path = str(parsed.get("resolution_path", "simple")).lower().strip()
    if path not in ("simple", "complex"):
        path = "simple"

    elapsed = (time.perf_counter() - started) * 1000
    log.info(
        "router.classified",
        extra={
            "intent": intent,
            "intent_confidence": confidence,
            "resolution_path": path,
            "provider": result.provider,
            "latency_ms": round(elapsed, 1),
            "pii_kinds": masked.kinds,
        },
    )

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "resolution_path": path,
        # Per-turn tool budget. Left cumulative it would grow with conversation
        # length and eventually trip `tool_attempts_exhausted` on a healthy chat.
        "tool_attempts": 0,
        # Complex paths get the frontier model; simple paths stay on the SLM.
        "model_tier": "frontier" if path == "complex" else "slm",
        "llm_provider": result.provider,
        "llm_stubbed": result.stubbed,
        "node_timings_ms": {"router": round(elapsed, 2)},
    }


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return 0.0
