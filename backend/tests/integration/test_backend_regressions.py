"""Regression tests for bugs that only appeared with real backends attached.

Every bug covered here passed the full suite while the system ran on in-process
fallbacks. They are the reason this file exists: each one is cheap to reintroduce
and expensive to notice, because none of them raise where you can see them.

See docs/DEVIATIONS.md section 16 for the incident write-ups.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.timeout(120)

BACKEND_ROOT = Path(__file__).resolve().parents[2]


# --- 1. Reserved logging keys ------------------------------------------
def test_reserved_extra_key_does_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    """`extra={"name": ...}` used to raise, and callers caught it as their own failure.

    `log.info("qdrant.collection_created", extra={"name": ...})` sat inside
    `QdrantHybridStore.connect`'s try block. The KeyError from logging was
    swallowed by `except Exception` and reported a healthy Qdrant as
    unavailable.
    """
    from app.utils.logging import get_logger

    log = get_logger("regression.reserved")
    with caplog.at_level(logging.INFO):
        # Every one of these is an existing LogRecord attribute.
        log.info("collision", extra={"name": "x", "module": "y", "args": "z", "levelname": "w"})

    assert caplog.records, "log call was swallowed"
    record = caplog.records[-1]
    # Renamed rather than dropped: the value must still reach the log.
    assert record.x_name == "x"
    assert record.name == "regression.reserved"  # the real logger name survives


def test_non_reserved_keys_are_untouched(caplog: pytest.LogCaptureFixture) -> None:
    from app.utils.logging import get_logger

    with caplog.at_level(logging.INFO):
        get_logger("regression.plain").info("ok", extra={"collection": "kb", "count": 3})

    record = caplog.records[-1]
    assert record.collection == "kb"
    assert record.count == 3


def test_reserved_set_is_derived_not_hand_listed() -> None:
    """A hand-maintained list drifts with the Python version and misses attributes."""
    from app.utils.logging import _RESERVED_ATTRS

    live = set(logging.LogRecord("n", 0, "p", 0, "m", None, None).__dict__)
    assert live <= _RESERVED_ATTRS, f"missing: {live - _RESERVED_ATTRS}"
    assert "taskName" in _RESERVED_ATTRS or sys.version_info < (3, 12)


def test_every_extra_key_in_the_app_is_safe() -> None:
    """Belt and braces: no real call site passes a reserved key.

    Parses the AST rather than scanning text, so the docstrings in
    `utils/logging.py` that *describe* the hazard are not mistaken for it.
    """
    import ast

    reserved = set(logging.LogRecord("n", 0, "p", 0, "m", None, None).__dict__)
    offenders: list[str] = []

    for path in (BACKEND_ROOT / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "extra" or not isinstance(keyword.value, ast.Dict):
                    continue
                for key in keyword.value.keys:
                    if isinstance(key, ast.Constant) and key.value in reserved:
                        offenders.append(f"{path.name}:{node.lineno} {key.value}")

    assert not offenders, f"reserved logging keys at call sites: {offenders}"


# --- 2. Qdrant point id stability --------------------------------------
def test_point_id_is_stable_within_a_process() -> None:
    from app.services.vector_store import point_id

    assert point_id("kb-refund-timing") == point_id("kb-refund-timing")


def test_point_id_is_stable_across_processes() -> None:
    """The actual bug: `hash()` is randomised per process by PYTHONHASHSEED.

    Ids differing between runs made every restart re-insert the whole corpus
    under fresh ids instead of overwriting it, so the collection grew and RRF
    started fusing duplicate copies of the same article.
    """
    script = (
        f"import sys; sys.path.insert(0, r'{BACKEND_ROOT}');"
        "from app.services.vector_store import point_id;"
        "print(point_id('kb-refund-timing'))"
    )
    seen = set()
    for seed in ("0", "1", "random"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
            env={"PATH": "", "SYSTEMROOT": "", "PYTHONHASHSEED": seed, **_min_env()},
        )
        seen.add(result.stdout.strip())

    assert len(seen) == 1, f"point id varies across processes: {seen}"


def test_point_id_never_uses_builtin_hash() -> None:
    """Guards the fix itself: `hash()` here would silently reintroduce the bug.

    Inspects the parsed function body, not its text -- the docstring explaining
    the hazard mentions `hash()` and must not trip this.
    """
    import ast

    source = (BACKEND_ROOT / "app" / "services" / "vector_store.py").read_text(encoding="utf-8")
    func = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "point_id"
    )
    called = {
        n.func.id
        for n in ast.walk(func)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "hash" not in called, "point_id must not use Python's randomised hash()"


def test_distinct_documents_get_distinct_ids() -> None:
    from app.ingestion.knowledge_loader import load_articles, to_documents
    from app.services.vector_store import point_id

    docs = to_documents(load_articles())
    ids = [point_id(d.id) for d in docs]
    assert len(ids) == len(set(ids)), "point id collision across the corpus"


async def test_upsert_is_idempotent(knowledge_store: Any) -> None:
    """Re-ingesting must overwrite, not accumulate -- the restart-growth bug."""
    from app.ingestion.knowledge_loader import ingest

    before = knowledge_store.describe()["documents"]
    await ingest(knowledge_store)
    await ingest(knowledge_store)
    assert knowledge_store.describe()["documents"] == before


async def test_retrieval_returns_no_duplicate_documents(knowledge_store: Any) -> None:
    """The visible symptom: one article cited five times."""
    hits = await knowledge_store.search("how long do refunds take", limit=5)
    ids = [h.id for h in hits]
    assert len(ids) == len(set(ids)), f"duplicate documents in results: {ids}"


# --- 3. Checkpointer connection lifetime -------------------------------
def test_checkpointer_keeps_a_reference_to_its_pool() -> None:
    """The bug: `from_conn_string()`'s context manager was entered and dropped.

    Garbage collection then ran `__aexit__` and closed the connection after
    `setup()` had already succeeded, so startup logged success and the first
    real query failed with "the connection is closed".
    """
    source = (BACKEND_ROOT / "app" / "agents" / "graph.py").read_text(encoding="utf-8")

    assert "_pg_pool" in source, "the pool must be held at module scope, not dropped"
    assert "AsyncConnectionPool" in source
    assert "__aenter__" not in source, (
        "entering a context manager without holding it is what caused the bug"
    )


def test_checkpointer_pool_is_closed_on_shutdown() -> None:
    from app.agents import graph

    assert hasattr(graph, "shutdown_checkpointer"), "pool must be released on shutdown"


def test_pool_connections_are_configured_for_langgraph() -> None:
    """AsyncPostgresSaver requires autocommit + dict_row; pooling needs prepare_threshold=0."""
    source = (BACKEND_ROOT / "app" / "agents" / "graph.py").read_text(encoding="utf-8")
    for required in ('"autocommit": True', '"prepare_threshold": 0', '"row_factory": dict_row'):
        assert required in source, f"missing pool connection setting: {required}"


# --- 4. Windows event loop ---------------------------------------------
def test_entry_point_forces_a_selector_loop() -> None:
    """psycopg cannot run async on Windows' ProactorEventLoop.

    uvicorn 0.40 picks the loop with a factory that hardcodes ProactorEventLoop
    on Windows and ignores `set_event_loop_policy`, so the fix has to bypass
    `uvicorn.run` and drive `Server.serve()` on an explicit loop.
    """
    source = (BACKEND_ROOT / "run.py").read_text(encoding="utf-8")
    assert "SelectorEventLoop" in source
    assert "asyncio.Runner" in source, "must drive the server on an explicit loop"


def test_loop_factory_returns_a_psycopg_compatible_loop() -> None:
    sys.path.insert(0, str(BACKEND_ROOT))
    from run import loop_factory

    loop = loop_factory()
    try:
        assert not isinstance(loop, getattr(asyncio, "ProactorEventLoop", ()))
    finally:
        loop.close()


def test_main_does_not_pretend_to_set_the_policy() -> None:
    """Setting a policy in app.main cannot work: uvicorn builds its loop first."""
    source = (BACKEND_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "set_event_loop_policy" not in source, (
        "a policy set here is silently ineffective; it belongs in run.py"
    )


# --- 5. Alembic driver normalisation -----------------------------------
def test_alembic_normalises_to_the_installed_driver() -> None:
    """`postgresql://` makes SQLAlchemy reach for psycopg2, which is not installed."""
    source = (BACKEND_ROOT / "alembic" / "env.py").read_text(encoding="utf-8")
    assert "postgresql+psycopg://" in source
    assert '.replace("postgresql+psycopg", "postgresql")' not in source, (
        "stripping the driver reintroduces the psycopg2 lookup"
    )


@pytest.mark.parametrize(
    ("given", "expected_driver"),
    [
        ("postgresql://u:p@h:5432/db", "postgresql+psycopg://"),
        ("postgres://u:p@h:5432/db", "postgresql+psycopg://"),
        ("postgresql+asyncpg://u:p@h:5432/db", "postgresql+psycopg://"),
        ("postgresql+psycopg://u:p@h:5432/db", "postgresql+psycopg://"),
    ],
)
def test_postgres_urls_resolve_to_psycopg3(given: str, expected_driver: str) -> None:
    resolved = _resolve_alembic_url(given)
    assert resolved.startswith(expected_driver), f"{given} -> {resolved}"


def test_sqlite_urls_lose_their_async_driver() -> None:
    assert _resolve_alembic_url("sqlite+aiosqlite:///./x.db") == "sqlite:///./x.db"


def _resolve_alembic_url(url: str) -> str:
    """Mirror of alembic/env.py `_sync_url` normalisation, minus the settings read."""
    url = url.replace("+aiosqlite", "")
    if url.startswith("postgres"):
        for async_driver in ("+asyncpg", "+psycopg_async"):
            url = url.replace(async_driver, "")
        if "+" not in url.split("://", 1)[0]:
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


# --- 6. Test isolation --------------------------------------------------
async def test_suite_does_not_bind_to_a_running_qdrant(knowledge_store: Any) -> None:
    """The suite must not change behaviour based on whether Docker is up.

    Qdrant has no "configured?" setting -- it always attempts a connection -- so
    with the client installed and a container running, the offline fixtures
    silently used the real server.
    """
    assert knowledge_store.describe()["backend"] == "in-memory"


def _min_env() -> dict[str, str]:
    import os

    keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "COMSPEC", "PATHEXT")
    return {k: os.environ[k] for k in keep if k in os.environ}


# --- 7. Dangling ticket references -------------------------------------
async def test_a_checkpoint_pointing_at_a_missing_ticket_recovers(
    compiled_graph: Any, crm: Any
) -> None:
    """Checkpoints are durable; the ticket store is a separate system.

    A conversation resumed from a checkpoint can carry a ticket_id that no
    longer exists. The update then silently no-opped, so the conversation kept
    referencing a ghost ticket for the rest of its life and no record was ever
    written again.
    """
    from langchain_core.messages import HumanMessage

    from app.agents.state import initial_state

    thread = "dangling-1"
    state = initial_state(customer_id="cust_demo", channel="webchat", conversation_id=thread)
    state["messages"] = [HumanMessage(content="How long do refunds take?")]
    state["ticket_id"] = "TKT-does-not-exist"

    result = await compiled_graph.ainvoke(
        dict(state), config={"configurable": {"thread_id": thread}}
    )

    assert result["ticket_id"] != "TKT-does-not-exist", "a ghost id was carried forward"
    recovered = await crm.get_ticket(result["ticket_id"])
    assert recovered is not None, "no ticket was created to replace the missing one"
    assert recovered.summary, "the replacement ticket was never populated"
