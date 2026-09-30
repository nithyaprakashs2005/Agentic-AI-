"""Authentication node and its routing gate.

system specification Graph Nodes #2: verifies identity, and gates the graph so no
resolution node runs while `authenticated` is False. Repeated failures escalate
to a human rather than looping forever.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from langchain_core.messages import AIMessage

from app.agents.state import SupportAgentState, last_user_text
from app.services.crm import get_crm
from app.utils.logging import get_logger

log = get_logger(__name__)

MAX_AUTH_ATTEMPTS = 3

_CHALLENGE = (
    "Before I can look at your account, I need to verify who I am speaking with. "
    "Could you confirm the email address or phone number on the account?"
)

_CHALLENGE_RETRY = (
    "Thanks, but I could not match that to an account. Could you try the other "
    "one -- either the email address or the phone number we have on file?"
)


async def authentication_node(state: SupportAgentState) -> dict[str, Any]:
    """Resolve the caller against the CRM, or issue a verification challenge."""
    started = time.perf_counter()

    if state.get("authenticated"):
        return {"node_timings_ms": {"auth": 0.0}}

    attempts = int(state.get("auth_attempts", 0))
    channel = state.get("channel", "webchat")
    customer_id = state.get("customer_id", "")

    crm = get_crm()

    # A verified channel identity (an authenticated web session, or a phone
    # number matched from caller ID) is sufficient on its own.
    profile = await crm.identify(
        customer_id=customer_id,
        utterance=last_user_text(state),
        channel=channel,
    )

    elapsed = round((time.perf_counter() - started) * 1000, 2)

    if profile is not None:
        log.info(
            "auth.verified",
            extra={
                "customer_id": profile.customer_id,
                "tier": profile.tier,
                "method": profile.matched_on,
                "attempts": attempts,
            },
        )
        return {
            "authenticated": True,
            "customer_id": profile.customer_id,
            "customer_tier": profile.tier,
            "node_timings_ms": {"auth": elapsed},
        }

    attempts += 1
    log.info("auth.challenge", extra={"attempts": attempts, "channel": channel})

    if attempts >= MAX_AUTH_ATTEMPTS:
        return {
            "auth_attempts": attempts,
            "authenticated": False,
            "escalation_required": True,
            "escalation_reason": "identity_verification_failed",
            "node_timings_ms": {"auth": elapsed},
        }

    prompt = _CHALLENGE if attempts == 1 else _CHALLENGE_RETRY
    return {
        "auth_attempts": attempts,
        "authenticated": False,
        "draft_response": prompt,
        "messages": [AIMessage(content=prompt)],
        "node_timings_ms": {"auth": elapsed},
    }


def check_authenticated(
    state: SupportAgentState,
) -> Literal["authenticated", "challenge", "escalate"]:
    """Routing gate out of the auth node."""
    if state.get("authenticated"):
        return "authenticated"
    if state.get("escalation_required") or int(state.get("auth_attempts", 0)) >= MAX_AUTH_ATTEMPTS:
        return "escalate"
    return "challenge"
