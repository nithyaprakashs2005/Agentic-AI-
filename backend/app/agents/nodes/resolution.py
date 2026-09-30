"""Resolution nodes for the simple and complex paths.

system specification wires `create_react_agent` for simple paths and `create_deep_agent`
for complex ones. Both prebuilt constructors require a LangChain
`BaseChatModel`; this deployment routes through a custom multi-provider gateway
with an offline stub so the system stays runnable with no API keys at all.
These nodes therefore reproduce the two *behaviours* over that gateway:

* `simple_resolution`  -- one grounded pass, with a bounded set of direct tool
  lookups chosen from the routed intent (ReAct-shaped, depth 1).
* `complex_resolution` -- writes an explicit plan, fans the plan steps out
  across MCP tools concurrently, then synthesises one grounded answer
  (Deep-Agent-shaped: plan, delegate, verify).

See docs/DEVIATIONS.md for the swap-in path back to the prebuilt constructors.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer

from app.agents.prompts.system import build_answer_messages
from app.agents.state import SupportAgentState, last_user_text
from app.mcp.tools import call_tool
from app.services.llm import get_gateway
from app.utils.logging import get_logger
from app.utils.pii import mask

log = get_logger(__name__)

# Which MCP tools each intent is allowed to reach. Keeps a FAQ lookup from
# touching billing, and bounds the blast radius of a misrouted turn.
_INTENT_TOOLS: dict[str, tuple[str, ...]] = {
    "orders": ("crm.list_orders",),
    "billing": ("billing.list_invoices",),
    "returns": ("crm.list_orders",),
    "technical": (),
    "general": (),
}

_COMPLEX_PLAN: dict[str, tuple[tuple[str, str], ...]] = {
    "billing": (
        ("billing.list_invoices", "Retrieve the full invoice history"),
        ("billing.detect_duplicates", "Check for duplicate charges"),
        ("crm.list_orders", "Cross-reference orders against those charges"),
        ("crm.get_customer", "Confirm the account tier and entitlements"),
    ),
    "orders": (
        ("crm.list_orders", "Retrieve all recent orders"),
        ("crm.get_customer", "Confirm the account tier and entitlements"),
    ),
    "returns": (
        ("crm.list_orders", "Identify the order being returned"),
        ("billing.list_invoices", "Locate the payment to refund"),
    ),
    "technical": (("crm.get_customer", "Confirm the account tier and entitlements"),),
    "general": (("crm.get_customer", "Confirm the account tier and entitlements"),),
}


async def simple_resolution_node(state: SupportAgentState) -> dict[str, Any]:
    """One grounded pass with shallow tool support."""
    started = time.perf_counter()
    question = last_user_text(state)
    intent = state.get("intent", "general")
    customer_id = state.get("customer_id", "")

    tool_names = _INTENT_TOOLS.get(intent, ())
    tool_results: dict[str, Any] = {}
    if tool_names and customer_id:
        gathered = await asyncio.gather(
            *(call_tool(name, {"customer_id": customer_id}) for name in tool_names)
        )
        tool_results = dict(zip(tool_names, gathered, strict=True))

    answer, result = await _generate_answer(
        state, question=question, tool_results=tool_results, plan=None
    )
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    log.info(
        "resolution.simple",
        extra={"intent": intent, "tools": list(tool_results), "latency_ms": elapsed},
    )

    return {
        "draft_response": answer,
        "messages": [AIMessage(content=answer)],
        "tool_attempts": int(state.get("tool_attempts", 0)) + len(tool_results),
        "resolution_status": "pending",
        "llm_provider": result.provider,
        "llm_stubbed": result.stubbed,
        "node_timings_ms": {"simple_resolution": elapsed},
    }


async def complex_resolution_node(state: SupportAgentState) -> dict[str, Any]:
    """Plan, fan out across MCP tools, then synthesise one grounded answer."""
    started = time.perf_counter()
    question = last_user_text(state)
    intent = state.get("intent", "general")
    customer_id = state.get("customer_id", "")

    plan = _COMPLEX_PLAN.get(intent, _COMPLEX_PLAN["general"])
    plan_steps = [description for _, description in plan]

    tool_results: dict[str, Any] = {}
    if customer_id:
        # Sub-agent-style parallel delegation: every plan step runs concurrently.
        gathered = await asyncio.gather(
            *(call_tool(name, {"customer_id": customer_id}) for name, _ in plan)
        )
        tool_results = {name: result for (name, _), result in zip(plan, gathered, strict=True)}

    answer, result = await _generate_answer(
        state, question=question, tool_results=tool_results, plan=plan_steps
    )
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    log.info(
        "resolution.complex",
        extra={
            "intent": intent,
            "plan_steps": len(plan_steps),
            "tools": list(tool_results),
            "latency_ms": elapsed,
        },
    )

    return {
        "draft_response": answer,
        "messages": [AIMessage(content=answer)],
        "tool_attempts": int(state.get("tool_attempts", 0)) + len(tool_results),
        "resolution_status": "pending",
        "deep_agent_todos": plan_steps,
        "llm_provider": result.provider,
        "llm_stubbed": result.stubbed,
        "node_timings_ms": {"complex_resolution": elapsed},
    }


async def _generate_answer(
    state: SupportAgentState,
    *,
    question: str,
    tool_results: dict[str, Any],
    plan: list[str] | None,
) -> tuple[str, Any]:
    """Build the grounded answer prompt and call the frontier tier."""
    context_parts = [state.get("rag_context", "")]

    if plan:
        context_parts.append(
            "Investigation plan carried out:\n" + "\n".join(f"- {step}" for step in plan)
        )

    for name, payload in tool_results.items():
        if payload.get("error"):
            # Tool faults are stated as unavailability, never as an error dump.
            context_parts.append(f"System data ({name}): unavailable for this request.")
        else:
            context_parts.append(f"System data ({name}): {payload}")

    context = "\n\n".join(part for part in context_parts if part and part.strip())

    messages = build_answer_messages(
        question=mask(question).text,
        context=context,
        channel=state.get("channel", "webchat"),
        customer_tier=state.get("customer_tier", "standard"),
        escalating=bool(state.get("escalation_required")),
    )

    result = await get_gateway().complete(
        messages,
        tier="frontier",
        purpose="answer",
        temperature=0.3,
        max_tokens=1200,  # headroom: thinking models spend budget before text
        on_delta=_answer_delta_writer(),
    )
    return result.text.strip(), result


def _answer_delta_writer() -> Callable[[str], None] | None:
    """Forward answer tokens to whoever is streaming this run, if anyone is.

    Only the customer-facing answer is streamed. Classification, sentiment and
    relevance calls also go through the gateway, and piping those to the client
    would leak the agent's internal reasoning into the chat window.

    Returns None outside a streaming context -- a plain `ainvoke`, a test, or
    any non-streaming caller -- so those paths stay buffered and unchanged.
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:
        # Raised when there is no runnable context at all -- a direct call from
        # a test or a script. Inside a run that simply is not streaming
        # `custom`, LangGraph hands back a no-op writer, which costs nothing.
        return None

    def emit(fragment: str) -> None:
        writer({"type": "answer_delta", "text": fragment})

    return emit
