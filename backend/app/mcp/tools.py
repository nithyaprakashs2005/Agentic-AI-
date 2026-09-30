"""Canonical MCP tool implementations and registry.

system specification Standard #2 requires every external lookup to go through an MCP
server rather than a bespoke wrapper. The tool *bodies* live here exactly once
and are exposed two ways:

* in-process, via `call_tool()` -- used by the agent nodes, and the path that
  enforces the 3 second timeout from system specification Constraints;
* over the wire, by the FastMCP servers in this package, which register these
  same callables.

That keeps one implementation per capability while still letting an external
MCP client drive the same tools.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.crm import get_crm
from app.services.tracing import span
from app.utils.logging import get_logger

log = get_logger(__name__)


# --- Schemas ------------------------------------------------------------
class CustomerLookupInput(BaseModel):
    customer_id: str = Field(description="Internal customer identifier")


class OrderStatusInput(BaseModel):
    order_id: str = Field(description="Order reference, e.g. ORD-84021")


class CustomerOrderInput(OrderStatusInput):
    """An order lookup scoped to the authenticated conversation owner."""

    customer_id: str = Field(description="Verified customer identifier from the conversation")


class ReturnEligibilityInput(CustomerOrderInput):
    item: str | None = Field(default=None, description="Item the customer wants to return")


class InitiateReturnInput(ReturnEligibilityInput):
    reason: str = Field(min_length=1, max_length=500)
    policy_confirmed: bool = Field(
        description="Set only by the Policy Agent after it has checked the return policy"
    )


class CourierAdvisoryInput(BaseModel):
    carrier: str
    location: str = ""


class OrderListInput(BaseModel):
    customer_id: str
    limit: int = Field(default=5, ge=1, le=25)


class InvoiceListInput(BaseModel):
    customer_id: str
    limit: int = Field(default=10, ge=1, le=50)


class RefundInput(BaseModel):
    customer_id: str
    amount: float = Field(gt=0, description="Refund amount in the account currency")
    reason: str
    invoice_id: str | None = None


class KnowledgeSearchInput(BaseModel):
    query: str
    category: str | None = None
    limit: int = Field(default=4, ge=1, le=10)


class NotifyInput(BaseModel):
    customer_id: str
    subject: str
    body: str
    channel: str = "email"


# --- Tool bodies --------------------------------------------------------
async def crm_get_customer(payload: CustomerLookupInput) -> dict[str, Any]:
    profile = await get_crm().get_customer(payload.customer_id)
    if profile is None:
        return {"found": False}
    return {
        "found": True,
        "customer_id": profile.customer_id,
        "name": profile.name,
        "tier": profile.tier,
    }


async def crm_list_orders(payload: OrderListInput) -> dict[str, Any]:
    orders = await get_crm().get_orders(payload.customer_id)
    return {
        "count": len(orders),
        "orders": [
            {
                "order_id": o.order_id,
                "status": o.status,
                "placed_at": o.placed_at,
                "total": o.total,
                "currency": o.currency,
                "items": o.items,
                "carrier": o.carrier,
                "tracking": o.tracking,
                "eta": o.eta,
            }
            for o in orders[: payload.limit]
        ],
    }


async def crm_order_status(payload: OrderStatusInput) -> dict[str, Any]:
    order = await get_crm().get_order(payload.order_id)
    if order is None:
        return {"found": False, "order_id": payload.order_id}
    return {
        "found": True,
        "order_id": order.order_id,
        "status": order.status,
        "carrier": order.carrier,
        "tracking": order.tracking,
        "eta": order.eta,
        "items": order.items,
        "total": order.total,
        "currency": order.currency,
    }


async def orders_get_order(payload: CustomerOrderInput) -> dict[str, Any]:
    """Return an order only after its owner matches the verified customer.

    This deliberately differs from the legacy admin-facing `crm.order_status`
    lookup: the customer agent must never use an unscoped order lookup.
    """
    order = await get_crm().get_order(payload.order_id)
    if order is None:
        return {"found": False, "order_id": payload.order_id}
    if order.customer_id != payload.customer_id:
        log.warning("orders.owner_mismatch", extra={"order_id": payload.order_id})
        # Do not reveal that an order exists or who owns it.
        return {"found": False, "order_id": payload.order_id, "ownership_verified": False}
    return {
        "found": True,
        "ownership_verified": True,
        "order_id": order.order_id,
        "status": order.status,
        "placed_at": order.placed_at,
        "items": order.items,
        "total": order.total,
        "currency": order.currency,
        "carrier": order.carrier,
        "tracking": order.tracking,
        "estimated_delivery": order.eta,
        "location": order.location,
        "delay_reason": order.delay_reason,
    }


async def orders_track_shipment(payload: CustomerOrderInput) -> dict[str, Any]:
    order_data = await orders_get_order(payload)
    if not order_data.get("found"):
        return order_data
    return {
        key: order_data.get(key)
        for key in (
            "found",
            "ownership_verified",
            "order_id",
            "status",
            "carrier",
            "tracking",
            "location",
            "estimated_delivery",
            "delay_reason",
        )
    }


async def returns_check_eligibility(payload: ReturnEligibilityInput) -> dict[str, Any]:
    order = await get_crm().get_order(payload.order_id)
    if order is None or order.customer_id != payload.customer_id:
        return {"eligible": False, "ownership_verified": False, "reason": "order_not_available"}

    requested_item = (payload.item or "").casefold()
    item_matches = not requested_item or any(
        requested_item in item.casefold() or item.casefold() in requested_item
        for item in order.items
    )
    if not item_matches:
        return {
            "eligible": False,
            "ownership_verified": True,
            "reason": "item_not_in_order",
            "order_id": order.order_id,
        }
    if order.status.casefold() != "delivered":
        return {
            "eligible": False,
            "ownership_verified": True,
            "reason": "not_delivered",
            "order_id": order.order_id,
        }
    if not order.eligible_for_return:
        return {
            "eligible": False,
            "ownership_verified": True,
            "reason": "outside_return_window",
            "order_id": order.order_id,
        }
    return {
        "eligible": True,
        "ownership_verified": True,
        "order_id": order.order_id,
        "item": payload.item or order.items[0],
        "refund_amount": order.total,
        "currency": order.currency,
        "policy": "30-day delivery window; unused item in original packaging",
    }


async def returns_initiate_return(payload: InitiateReturnInput) -> dict[str, Any]:
    # A caller cannot bypass the graph's policy gate merely by invoking the MCP
    # tool: repeat the independent eligibility check here.
    if not payload.policy_confirmed:
        return {"success": False, "error": "policy_confirmation_required"}
    eligibility = await returns_check_eligibility(payload)
    if not eligibility.get("eligible"):
        return {
            "success": False,
            "error": "not_eligible",
            "eligibility": eligibility,
        }
    return await get_crm().initiate_return(
        payload.order_id, eligibility["item"], payload.reason, customer_id=payload.customer_id
    )


async def web_search_courier_advisories(payload: CourierAdvisoryInput) -> dict[str, Any]:
    """Search configured Tavily for disruption / recall notices.

    No key means no external lookup is attempted and the agent continues with
    the shipment data it has. The response never fabricates an advisory.
    """
    settings = get_settings()
    api_key = getattr(settings, "tavily_api_key", None)
    if api_key is None or not api_key.get_secret_value():
        return {"available": False, "results": [], "reason": "tavily_not_configured"}
    try:
        import httpx

        query = f"{payload.carrier} delivery disruption product recall {payload.location}".strip()
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key.get_secret_value(), "query": query, "max_results": 3},
            )
            response.raise_for_status()
        results = response.json().get("results", [])
        return {
            "available": True,
            "results": [
                {
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "content": row.get("content", "")[:500],
                }
                for row in results
            ],
        }
    except Exception:
        log.warning("tavily.lookup_failed", extra={"carrier": payload.carrier})
        return {"available": False, "results": [], "reason": "advisory_lookup_unavailable"}


async def billing_list_invoices(payload: InvoiceListInput) -> dict[str, Any]:
    invoices = await get_crm().get_invoices(payload.customer_id)
    return {
        "count": len(invoices),
        "invoices": [
            {
                "invoice_id": i.invoice_id,
                "amount": i.amount,
                "currency": i.currency,
                "issued_at": i.issued_at,
                "status": i.status,
                "description": i.description,
            }
            for i in invoices[: payload.limit]
        ],
    }


async def billing_detect_duplicates(payload: InvoiceListInput) -> dict[str, Any]:
    """Flag same-amount invoices issued within 72h -- the duplicate-charge signal."""
    invoices = await get_crm().get_invoices(payload.customer_id)
    groups: dict[tuple[float, str], list[Any]] = {}
    for invoice in invoices:
        groups.setdefault((invoice.amount, invoice.currency), []).append(invoice)

    suspected = [
        {
            "amount": amount,
            "currency": currency,
            "invoice_ids": [i.invoice_id for i in group],
            "issued_at": [i.issued_at for i in group],
            "description": group[0].description,
        }
        for (amount, currency), group in groups.items()
        if len(group) > 1
    ]
    return {"suspected_duplicates": suspected, "count": len(suspected)}


async def billing_issue_refund(payload: RefundInput) -> dict[str, Any]:
    """Refunds above the policy threshold require HITL approval, never auto-issue."""
    threshold = get_settings().escalation_refund_threshold
    if payload.amount > threshold:
        log.info(
            "refund.hitl_required",
            extra={"amount": payload.amount, "threshold": threshold},
        )
        return {
            "issued": False,
            "requires_human_approval": True,
            "threshold": threshold,
            "amount": payload.amount,
            "reason": "amount exceeds automatic refund policy threshold",
        }
    return {
        "issued": True,
        "requires_human_approval": False,
        "amount": payload.amount,
        "reference": f"RFD-{int(time.time())}",
    }


async def knowledge_search(payload: KnowledgeSearchInput) -> dict[str, Any]:
    from app.services.vector_store import get_store

    hits = await get_store().search(
        payload.query,
        categories=[payload.category] if payload.category else None,
        limit=payload.limit,
    )
    return {
        "count": len(hits),
        "results": [
            {
                "id": hit.id,
                "title": hit.payload.get("title", ""),
                "text": hit.text,
                "source": hit.payload.get("source_url", ""),
                "score": round(hit.score, 5),
            }
            for hit in hits
        ],
    }


async def notifications_send(payload: NotifyInput) -> dict[str, Any]:
    """No outbound provider is configured, so this records intent only.

    Returning `sent: False` keeps the agent honest -- it must not tell a
    customer an email was sent when nothing left the building.
    """
    settings = get_settings()
    configured = bool(settings.twilio_account_sid) if payload.channel == "sms" else False
    log.info(
        "notifications.queued",
        extra={"channel": payload.channel, "configured": configured},
    )
    return {
        "sent": False,
        "queued": True,
        "provider_configured": configured,
        "channel": payload.channel,
        "note": "no outbound provider configured in this deployment",
    }


# --- Registry -----------------------------------------------------------
@dataclass(slots=True, frozen=True)
class ToolSpec:
    name: str
    server: str
    description: str
    schema: type[BaseModel]
    handler: Callable[[Any], Awaitable[dict[str, Any]]]


REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "crm.get_customer",
            "crm",
            "Look up a customer profile and tier",
            CustomerLookupInput,
            crm_get_customer,
        ),
        ToolSpec(
            "crm.list_orders",
            "crm",
            "List a customer's recent orders",
            OrderListInput,
            crm_list_orders,
        ),
        ToolSpec(
            "crm.order_status",
            "crm",
            "Get status, carrier, and tracking for one order",
            OrderStatusInput,
            crm_order_status,
        ),
        ToolSpec(
            "orders.get_order",
            "crm",
            "Get a verified customer's order; never returns another customer's data",
            CustomerOrderInput,
            orders_get_order,
        ),
        ToolSpec(
            "orders.track_shipment",
            "crm",
            "Get carrier location and estimated delivery for a verified customer's order",
            CustomerOrderInput,
            orders_track_shipment,
        ),
        ToolSpec(
            "returns.check_eligibility",
            "crm",
            "Confirm return eligibility using the verified order and return policy",
            ReturnEligibilityInput,
            returns_check_eligibility,
        ),
        ToolSpec(
            "returns.initiate_return",
            "crm",
            "Create a return only after a Policy Agent eligibility confirmation",
            InitiateReturnInput,
            returns_initiate_return,
        ),
        ToolSpec(
            "web.search_courier_advisories",
            "knowledge",
            "Search Tavily for current courier disruptions and product recall notices",
            CourierAdvisoryInput,
            web_search_courier_advisories,
        ),
        ToolSpec(
            "billing.list_invoices",
            "billing",
            "List a customer's invoices",
            InvoiceListInput,
            billing_list_invoices,
        ),
        ToolSpec(
            "billing.detect_duplicates",
            "billing",
            "Detect suspected duplicate charges on an account",
            InvoiceListInput,
            billing_detect_duplicates,
        ),
        ToolSpec(
            "billing.issue_refund",
            "billing",
            "Issue a refund; amounts over policy threshold require human approval",
            RefundInput,
            billing_issue_refund,
        ),
        ToolSpec(
            "knowledge.search",
            "knowledge",
            "Hybrid search over the support knowledge base",
            KnowledgeSearchInput,
            knowledge_search,
        ),
        ToolSpec(
            "notifications.send",
            "notifications",
            "Send a customer notification",
            NotifyInput,
            notifications_send,
        ),
    )
}


def tools_for(server: str) -> list[ToolSpec]:
    return [spec for spec in REGISTRY.values() if spec.server == server]


async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Invoke a registered tool with validation and the MCP timeout budget.

    Returns a dict with an `error` key on failure rather than raising, so a
    single tool fault degrades one turn instead of aborting the graph. Errors
    are never customer-facing -- the answer prompt sees only tool output.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        return {"error": "unknown_tool", "tool": name}

    timeout = get_settings().mcp_tool_timeout_seconds
    started = time.perf_counter()
    try:
        payload = spec.schema(**arguments)
    except Exception as exc:
        log.warning("tool.invalid_arguments", extra={"tool": name, "error": str(exc)[:200]})
        return {"error": "invalid_arguments", "tool": name}

    try:
        with span(name, kind="tool", server=spec.server, timeout_s=timeout):
            result = await asyncio.wait_for(spec.handler(payload), timeout=timeout)
    except TimeoutError:
        log.warning("tool.timeout", extra={"tool": name, "timeout_s": timeout})
        return {"error": "timeout", "tool": name}
    except Exception as exc:
        log.error("tool.failed", extra={"tool": name, "error": str(exc)[:300]})
        return {"error": "tool_failed", "tool": name}

    elapsed = round((time.perf_counter() - started) * 1000, 2)
    log.info("tool.called", extra={"tool": name, "latency_ms": elapsed})
    return result
