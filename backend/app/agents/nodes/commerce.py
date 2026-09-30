"""Retail order and return nodes.

The generic resolution nodes are useful for FAQs, but commerce actions need a
stricter path: every order lookup is owner-scoped and a return can only be
created after the policy gate has produced an eligible decision.
"""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.messages import AIMessage

from app.agents.state import SupportAgentState, last_user_text
from app.mcp.tools import call_tool
from app.utils.logging import get_logger

log = get_logger(__name__)

_ORDER_ID = re.compile(r"(?:order\s*(?:number)?\s*|#)(ORD[-\s]?)?([A-Z0-9]{4,})", re.I)
_IGNORED_ITEM_WORDS = {
    "i", "want", "to", "return", "the", "my", "bought", "last", "week", "please",
    "order", "item", "items", "a", "an", "this", "that", "for", "because", "is",
}


def extract_order_id(text: str) -> str | None:
    """Extract a customer-provided identifier without guessing arbitrary digits."""
    match = _ORDER_ID.search(text.upper())
    if not match:
        return None
    prefix, body = match.groups()
    return f"ORD-{body}" if prefix else body


async def policy_agent_node(state: SupportAgentState) -> dict[str, Any]:
    """Confirm return eligibility before the Order Agent can create a return."""
    started = time.perf_counter()
    text = last_user_text(state)
    customer_id = state.get("customer_id", "")
    order_id = extract_order_id(text)

    # The product name is sufficient only if it identifies exactly one order
    # belonging to the authenticated customer. It must never search globally.
    if not order_id:
        listing = await call_tool("crm.list_orders", {"customer_id": customer_id, "limit": 25})
        order_id = _match_order_from_item(text, listing.get("orders", []))

    if not order_id:
        return {
            "return_eligibility": {"eligible": False, "reason": "order_reference_needed"},
            "policy_confirmed": False,
            "node_timings_ms": {"policy_agent": _elapsed(started)},
        }

    eligibility = await call_tool(
        "returns.check_eligibility",
        {"customer_id": customer_id, "order_id": order_id, "item": _requested_item(text)},
    )
    return {
        "commerce_order_id": order_id,
        "return_eligibility": eligibility,
        "policy_confirmed": bool(eligibility.get("eligible")),
        "tool_attempts": int(state.get("tool_attempts", 0)) + 1,
        "node_timings_ms": {"policy_agent": _elapsed(started)},
    }


async def order_agent_node(state: SupportAgentState) -> dict[str, Any]:
    """Resolve shipment questions and policy-approved returns using MCP tools."""
    started = time.perf_counter()
    intent = state.get("intent")
    text = last_user_text(state)
    customer_id = state.get("customer_id", "")
    attempts = int(state.get("tool_attempts", 0))

    if intent == "returns":
        eligibility = state.get("return_eligibility") or {}
        if eligibility.get("reason") == "order_reference_needed":
            answer = (
                "I can help start that return. Please share the order number so I can "
                "confirm eligibility."
            )
        elif not eligibility.get("eligible"):
            answer = _return_ineligible_reply(eligibility)
        else:
            result = await call_tool(
                "returns.initiate_return",
                {
                    "customer_id": customer_id,
                    "order_id": state.get("commerce_order_id", ""),
                    "item": eligibility.get("item"),
                    "reason": _return_reason(text),
                    "policy_confirmed": bool(state.get("policy_confirmed")),
                },
            )
            attempts += 1
            if result.get("success"):
                answer = (
                    f"Your return is confirmed for {result['item']} from "
                    f"order {result['order_id']}. "
                    f"Your return reference is {result['return_id']}. A prepaid label is ready at "
                    f"{result['label_url']}. {result['instructions']}"
                )
            else:
                # A duplicate tool-side eligibility check is a safety net. Its
                # generic failure is intentionally not presented as a promise.
                answer = (
                    "I could not create the return automatically. I have not made "
                    "any refund promise."
                )
        return _resolution(answer, started, attempts, "order_agent")

    order_id = extract_order_id(text)
    if not order_id:
        return _resolution(
            "Please share the order number and I will check its latest shipment status.",
            started,
            attempts,
            "order_agent",
        )

    shipment = await call_tool(
        "orders.track_shipment", {"customer_id": customer_id, "order_id": order_id}
    )
    attempts += 1
    if not shipment.get("found"):
        return _resolution(
            "I could not find that order in this verified account. Please check the "
            "order number or sign in to the account used for the purchase.",
            started,
            attempts,
            "order_agent",
        )

    advisory: dict[str, Any] = {}
    if "delayed" in str(shipment.get("status", "")).casefold():
        advisory = await call_tool(
            "web.search_courier_advisories",
            {
                "carrier": shipment.get("carrier", "carrier"),
                "location": shipment.get("location", ""),
            },
        )
        attempts += 1
    answer = _shipment_reply(shipment, advisory)
    return _resolution(answer, started, attempts, "order_agent")


def _match_order_from_item(text: str, orders: list[dict[str, Any]]) -> str | None:
    words = {
        word
        for word in re.findall(r"[a-z]{3,}", text.casefold())
        if word not in _IGNORED_ITEM_WORDS
    }
    candidates = [
        order for order in orders
        if any(word in " ".join(order.get("items", [])).casefold() for word in words)
    ]
    return candidates[0].get("order_id") if len(candidates) == 1 else None


def _requested_item(text: str) -> str | None:
    # Item matching is deliberately permissive only after the candidate order
    # is already owner-scoped; the eligibility tool verifies membership.
    words = [
        word
        for word in re.findall(r"[a-z]{3,}", text.casefold())
        if word not in _IGNORED_ITEM_WORDS
    ]
    return " ".join(words) or None


def _return_reason(text: str) -> str:
    return text.strip()[:500] or "Customer requested a return"


def _return_ineligible_reply(eligibility: dict[str, Any]) -> str:
    reason = eligibility.get("reason")
    if reason == "outside_return_window":
        return (
            "This order is outside the 30-day return window, so I cannot create a "
            "return under the standard policy."
        )
    if reason == "not_delivered":
        return "I can create a return after the item is delivered; it is not marked delivered yet."
    if reason == "item_not_in_order":
        return (
            "That item does not appear on the verified order. Please check the item "
            "name or order number."
        )
    return "I could not verify that order for this conversation, so I cannot access or change it."


def _shipment_reply(shipment: dict[str, Any], advisory: dict[str, Any]) -> str:
    order = shipment["order_id"]
    status = shipment.get("status", "Unknown")
    carrier = shipment.get("carrier") or "the carrier"
    location = shipment.get("location")
    eta = shipment.get("estimated_delivery")
    parts = [f"Order {order} is {status} with {carrier}."]
    if location:
        parts.append(f"Latest location: {location}.")
    if eta:
        parts.append(f"Estimated delivery: {eta}.")
    if shipment.get("delay_reason"):
        parts.append(f"Carrier update: {shipment['delay_reason']}.")
    if advisory.get("available") and advisory.get("results"):
        parts.append("I also found a current carrier advisory that may be relevant.")
    return " ".join(parts)


def _resolution(answer: str, started: float, attempts: int, node_name: str) -> dict[str, Any]:
    elapsed = _elapsed(started)
    log.info("commerce.resolved", extra={"node": node_name, "tool_attempts": attempts})
    return {
        "draft_response": answer,
        "messages": [AIMessage(content=answer)],
        "tool_attempts": attempts,
        "resolution_status": "resolved",
        "node_timings_ms": {node_name: elapsed},
    }


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
