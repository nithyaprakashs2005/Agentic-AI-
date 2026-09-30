"""MCP tool registry: validation, timeouts, and the refund HITL threshold."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from app.config import Settings
from app.mcp.tools import REGISTRY, call_tool, tools_for


def test_registry_covers_every_declared_server() -> None:
    """system specification names four MCP servers; each must expose at least one tool."""
    for server in ("crm", "billing", "knowledge", "notifications"):
        assert tools_for(server), f"no tools registered for {server}"


def test_tool_names_are_namespaced() -> None:
    for name in REGISTRY:
        assert "." in name, f"{name} is not namespaced server.tool"


async def test_unknown_tool_returns_an_error_not_an_exception() -> None:
    result = await call_tool("nope.missing", {})
    assert result["error"] == "unknown_tool"


async def test_invalid_arguments_are_rejected(crm: Any) -> None:
    result = await call_tool("crm.list_orders", {"wrong_field": 1})
    assert result["error"] == "invalid_arguments"


async def test_negative_refund_is_rejected_by_the_schema(crm: Any) -> None:
    result = await call_tool(
        "billing.issue_refund", {"customer_id": "cust_demo", "amount": -5, "reason": "x"}
    )
    assert result["error"] == "invalid_arguments"


async def test_order_lookup_returns_tracking(crm: Any) -> None:
    result = await call_tool("crm.order_status", {"order_id": "ORD-84021"})
    assert result["found"] is True
    assert result["status"] == "Shipped"
    assert result["tracking"]


async def test_order_lookup_is_case_insensitive(crm: Any) -> None:
    result = await call_tool("crm.order_status", {"order_id": "ord-84021"})
    assert result["found"] is True


async def test_customer_order_tools_do_not_disclose_another_customers_order(crm: Any) -> None:
    result = await call_tool(
        "orders.get_order", {"customer_id": "cust_demo", "order_id": "ORD-90112"}
    )
    assert result["found"] is False
    assert result["ownership_verified"] is False
    assert "items" not in result


async def test_return_requires_policy_confirmation_and_owner_check(crm: Any) -> None:
    blocked = await call_tool(
        "returns.initiate_return",
        {
            "customer_id": "cust_demo",
            "order_id": "ORD-10245",
            "item": "shoes",
            "reason": "does not fit",
            "policy_confirmed": False,
        },
    )
    assert blocked["success"] is False
    assert blocked["error"] == "policy_confirmation_required"

    eligible = await call_tool(
        "returns.check_eligibility",
        {"customer_id": "cust_demo", "order_id": "ORD-10245", "item": "shoes"},
    )
    assert eligible["eligible"] is True

    created = await call_tool(
        "returns.initiate_return",
        {
            "customer_id": "cust_demo",
            "order_id": "ORD-10245",
            "item": "shoes",
            "reason": "does not fit",
            "policy_confirmed": True,
        },
    )
    assert created["success"] is True
    assert created["return_id"].startswith("RET-")


async def test_missing_order_reports_not_found(crm: Any) -> None:
    result = await call_tool("crm.order_status", {"order_id": "ORD-00000"})
    assert result["found"] is False
    assert "error" not in result


async def test_duplicate_detection_finds_the_seeded_pair(crm: Any) -> None:
    """cust_abc123 has two identical 247.50 invoices two days apart."""
    result = await call_tool("billing.detect_duplicates", {"customer_id": "cust_abc123"})
    assert result["count"] == 1
    assert result["suspected_duplicates"][0]["amount"] == 247.50
    assert len(result["suspected_duplicates"][0]["invoice_ids"]) == 2


async def test_no_duplicates_for_distinct_amounts(crm: Any) -> None:
    result = await call_tool("billing.detect_duplicates", {"customer_id": "cust_demo"})
    assert result["count"] == 0


# --- Refund threshold: the HITL boundary --------------------------------
@pytest.mark.parametrize(
    ("amount", "expect_auto"),
    [
        (99.99, True),  # below the 100.00 threshold
        (100.00, True),  # exactly at it: strict greater-than
        (100.01, False),  # above it -> human approval
        (5000.0, False),
    ],
)
async def test_refund_threshold_boundary(
    amount: float, expect_auto: bool, crm: Any, _offline_settings: Settings
) -> None:
    result = await call_tool(
        "billing.issue_refund",
        {"customer_id": "cust_demo", "amount": amount, "reason": "duplicate charge"},
    )
    assert result["issued"] is expect_auto
    assert result["requires_human_approval"] is (not expect_auto)


async def test_notifications_never_claim_a_send_without_a_provider(crm: Any) -> None:
    """Telling a customer an email was sent when none was is a lie we must not tell."""
    result = await call_tool(
        "notifications.send",
        {"customer_id": "cust_demo", "subject": "s", "body": "b", "channel": "email"},
    )
    assert result["sent"] is False
    assert result["queued"] is True


async def test_knowledge_search_tool_returns_citable_sources(knowledge_store: Any) -> None:
    result = await call_tool("knowledge.search", {"query": "refund timing", "limit": 3})
    assert result["count"] > 0
    assert all(r["source"] for r in result["results"])


# --- Timeout ------------------------------------------------------------
async def test_tool_timeout_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """system specification Constraints: MCP calls are capped at 3s (configurable)."""
    import app.mcp.tools as tools_module

    async def never_returns(_payload: object) -> dict[str, Any]:
        await asyncio.sleep(10)
        return {}

    # ToolSpec is frozen by design; swap the registry entry instead.
    slow_spec = replace(REGISTRY["crm.get_customer"], handler=never_returns)
    monkeypatch.setitem(REGISTRY, "crm.get_customer", slow_spec)

    fast = Settings(_env_file=None, mcp_tool_timeout_seconds=0.05)  # type: ignore[call-arg]
    monkeypatch.setattr(tools_module, "get_settings", lambda: fast)

    result = await call_tool("crm.get_customer", {"customer_id": "cust_demo"})
    assert result["error"] == "timeout"


async def test_tool_exception_becomes_data(monkeypatch: pytest.MonkeyPatch) -> None:
    async def explodes(_payload: object) -> dict[str, Any]:
        raise RuntimeError("upstream on fire")

    monkeypatch.setitem(
        REGISTRY, "crm.get_customer", replace(REGISTRY["crm.get_customer"], handler=explodes)
    )
    result = await call_tool("crm.get_customer", {"customer_id": "cust_demo"})
    assert result["error"] == "tool_failed"
    # The internal message must not travel with it.
    assert "on fire" not in str(result)
