"""Main orchestrator: a custom LangGraph `StateGraph`.

Flow (system specification Graph Construction, with the inconsistencies resolved):

    START -> router -> auth
      auth --(authenticated)--> agentic_rag
      auth --(challenge)------> END          # ask, then wait for the next turn
      auth --(escalate)-------> escalation
      agentic_rag --(resolution_path)--> simple_resolution | complex_resolution
      *_resolution -> sentiment
      sentiment --(escalate)--> escalation
      sentiment --(continue)--> ticketing
      escalation -> ticketing
      ticketing --(handoff)--> human_handoff -> END     # dynamic interrupt()
      ticketing --(done)-----> END

Two deliberate departures from the system specification snippet, both fixing real defects:

* The `challenge` branch ends the turn instead of looping back into `auth`. A
  self-loop would spin without new customer input; the challenge question is
  already queued as the reply, so the next inbound turn re-enters the graph.
* The HITL pause sits after `ticketing`, so a CRM record exists before the run
  parks. See `nodes/escalation.py` for why the pause needs its own node.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.nodes.auth import authentication_node, check_authenticated
from app.agents.nodes.commerce import order_agent_node, policy_agent_node
from app.agents.nodes.escalation import escalation_node, human_handoff_node
from app.agents.nodes.resolution import complex_resolution_node, simple_resolution_node
from app.agents.nodes.retrieval import agentic_rag_node
from app.agents.nodes.router import router_node
from app.agents.nodes.sentiment import check_escalation, sentiment_eval_node
from app.agents.nodes.ticketing import route_after_ticketing, ticketing_node
from app.agents.state import SupportAgentState
from app.config import Settings, get_settings
from app.services.tracing import span
from app.utils.logging import get_logger

log = get_logger(__name__)


def route_by_resolution_path(state: SupportAgentState) -> str:
    """Read the path the router chose. This edge decides nothing itself."""
    return "complex" if state.get("resolution_path") == "complex" else "simple"


def route_after_retrieval(state: SupportAgentState) -> str:
    """Send commerce intents through their safe policy/action path."""
    if state.get("intent") == "returns":
        return "policy"
    if state.get("intent") == "orders":
        return "orders"
    return route_by_resolution_path(state)


def _traced(name: str, node: Any) -> Any:
    """Wrap a node so every one emits a span.

    Applied centrally at registration rather than by decorating each node, so a
    node added later cannot silently escape instrumentation.
    """

    async def run(state: SupportAgentState) -> Any:
        with span(name, kind="node", intent=state.get("intent"), channel=state.get("channel")) as s:
            update = await node(state)
            if isinstance(update, dict):
                s.metadata["keys"] = sorted(update)
            return update

    run.__name__ = f"traced_{name}"
    return run


def _sync_traced(name: str, node: Any) -> Any:
    """Span wrapper for synchronous nodes (the HITL gate raises interrupt())."""

    def run(state: SupportAgentState) -> Any:
        with span(name, kind="node", channel=state.get("channel")):
            return node(state)

    run.__name__ = f"traced_{name}"
    return run


def build_graph() -> StateGraph[SupportAgentState, Any, Any, Any]:
    """Assemble the orchestrator (uncompiled, so tests can inspect it)."""
    graph = StateGraph(SupportAgentState)

    graph.add_node("router", _traced("router", router_node))
    graph.add_node("auth", _traced("auth", authentication_node))
    graph.add_node("agentic_rag", _traced("agentic_rag", agentic_rag_node))
    graph.add_node("policy_agent", _traced("policy_agent", policy_agent_node))
    graph.add_node("order_agent", _traced("order_agent", order_agent_node))
    graph.add_node("simple_resolution", _traced("simple_resolution", simple_resolution_node))
    graph.add_node("complex_resolution", _traced("complex_resolution", complex_resolution_node))
    graph.add_node("sentiment", _traced("sentiment", sentiment_eval_node))
    graph.add_node("escalation", _traced("escalation", escalation_node))
    graph.add_node("ticketing", _traced("ticketing", ticketing_node))
    graph.add_node("human_handoff", _sync_traced("human_handoff", human_handoff_node))

    graph.add_edge(START, "router")
    graph.add_edge("router", "auth")

    # Authentication gate: nothing downstream runs unauthenticated.
    graph.add_conditional_edges(
        "auth",
        check_authenticated,
        {
            "authenticated": "agentic_rag",
            "challenge": END,
            "escalate": "escalation",
        },
    )

    graph.add_conditional_edges(
        "agentic_rag",
        route_after_retrieval,
        {
            "policy": "policy_agent",
            "orders": "order_agent",
            "simple": "simple_resolution",
            "complex": "complex_resolution",
        },
    )
    # The policy gate is intentionally separate. A return can only be created
    # by Order Agent after this node has recorded an eligible decision.
    graph.add_edge("policy_agent", "order_agent")
    graph.add_edge("order_agent", "sentiment")

    graph.add_edge("simple_resolution", "sentiment")
    graph.add_edge("complex_resolution", "sentiment")

    graph.add_conditional_edges(
        "sentiment",
        check_escalation,
        {"escalate": "escalation", "continue": "ticketing"},
    )

    graph.add_edge("escalation", "ticketing")

    graph.add_conditional_edges(
        "ticketing",
        route_after_ticketing,
        {"handoff": "human_handoff", "done": END},
    )
    graph.add_edge("human_handoff", END)

    return graph


_pg_pool: Any = None
"""Module-level so the pool outlives `build_checkpointer` and is closable."""


async def build_checkpointer(settings: Settings) -> tuple[Any, str]:
    """`AsyncPostgresSaver` on a connection pool, else `InMemorySaver`.

    A pool rather than `from_conn_string()` for two reasons:

    1. `from_conn_string()` returns a context manager. Entering it and dropping
       the reference -- which the previous implementation did -- lets it be
       garbage collected, and `__aexit__` then closes the connection. `setup()`
       succeeded, so the tables appeared and it looked healthy, but the first
       real query failed with "the connection is closed".
    2. A single connection serialises every checkpoint read and write across
       all concurrent conversations. A pool is what a server needs.

    `autocommit` and `dict_row` are required by AsyncPostgresSaver;
    `prepare_threshold=0` avoids prepared-statement reuse errors across pooled
    connections.
    """
    global _pg_pool

    if settings.has_postgres:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool

            _pg_pool = AsyncConnectionPool(
                conninfo=str(settings.database_url),
                min_size=1,
                max_size=10,
                open=False,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                },
            )
            await _pg_pool.open(wait=True, timeout=10)

            saver = AsyncPostgresSaver(_pg_pool)
            await saver.setup()
            log.info("checkpointer.postgres_ready", extra={"pool_max": 10})
            return saver, "postgres"
        except Exception as exc:
            log.warning("checkpointer.postgres_failed", extra={"error": str(exc)[:300]})
            if _pg_pool is not None:
                await _pg_pool.close()
                _pg_pool = None

    from langgraph.checkpoint.memory import InMemorySaver

    log.info("checkpointer.in_memory")
    return InMemorySaver(), "in-memory"


_compiled: Any = None
_checkpointer_kind = "uninitialised"


async def init_graph(settings: Settings | None = None) -> Any:
    """Compile the orchestrator once, at application startup."""
    global _compiled, _checkpointer_kind
    settings = settings or get_settings()
    checkpointer, kind = await build_checkpointer(settings)
    _checkpointer_kind = kind

    # No `interrupt_before`: the HITL pause is the dynamic interrupt() inside
    # human_handoff_node. Declaring both would pause twice for one handoff.
    _compiled = build_graph().compile(checkpointer=checkpointer)
    log.info("graph.compiled", extra={"checkpointer": kind})
    return _compiled


def get_graph() -> Any:
    if _compiled is None:
        raise RuntimeError("graph not initialised; call init_graph() at startup")
    return _compiled


def checkpointer_kind() -> str:
    return _checkpointer_kind


async def shutdown_checkpointer() -> None:
    """Release the PostgreSQL pool. Called from the app lifespan."""
    global _pg_pool
    if _pg_pool is not None:
        await _pg_pool.close()
        _pg_pool = None
        log.info("checkpointer.pool_closed")
