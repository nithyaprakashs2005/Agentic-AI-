"""Health, readiness, and degradation reporting.

`GET /health` is the honest inventory of what is actually live. Because this
deployment degrades rather than refusing to boot, that report is the only way to
tell a fully-wired install from an offline one -- so it names every fallback in
use instead of just saying "ok".
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter

from app.agents.graph import checkpointer_kind
from app.config import get_settings
from app.services.analytics import get_metrics
from app.services.crm import get_crm
from app.services.llm import get_gateway
from app.services.session import get_sessions
from app.services.tracing import get_tracer
from app.services.vector_store import get_store

router = APIRouter(tags=["health"])

# A probe exists to answer "is it up *now*". Anything slower than this is, for
# the purposes of serving a request, already down.
PROBE_TIMEOUT_SECONDS = 2.0


async def _reachable(probe: Callable[[], Awaitable[Any]]) -> bool:
    """Run one liveness check, treating slow and broken as the same answer."""
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_SECONDS):
            await probe()
    except Exception:
        return False
    return True


def _crm_report(database_reachable: bool | None) -> dict[str, Any]:
    """CRM inventory, with durability claimed only when it is actually true.

    `describe()` reports `tickets_durable: true` because a database was wired
    at startup. Left alone, that sits next to `reachable: false` and flatly
    contradicts it -- writes are going to memory and will be lost.
    """
    from app.services.repository import get_repository

    report = dict(get_crm().describe())
    report["reachable"] = database_reachable
    if database_reachable is False:
        report["tickets_durable"] = False

    # getattr rather than a direct call: health reporting must not be the thing
    # that breaks when a backend does not implement an optional detail.
    state = getattr(get_repository(), "circuit_state", None)
    if callable(state):
        report["circuit"] = state()
    return report


async def _probe_subsystems() -> dict[str, bool]:
    """Ping every backing store at once, so /health stays fast.

    This reports what is live, not what was chosen at boot. The distinction is
    not academic: with every container stopped, this endpoint previously still
    reported `crm.backend: postgres`, `tickets_durable: true`, `sessions: redis`
    and an overall status of healthy-but-for-Twilio. A readiness probe that
    answers "ready" while its database is unreachable keeps a dead instance in
    the load balancer.
    """
    from app.services.repository import get_repository

    probes: dict[str, Callable[[], Awaitable[Any]]] = {}

    try:
        store = get_store()
        probes["knowledge"] = store.ping
    except RuntimeError:
        pass

    try:
        sessions = get_sessions()
        probes["sessions"] = sessions.ping
    except RuntimeError:
        pass

    repo = get_repository()
    if repo is not None:
        probes["database"] = repo.count_tickets

    if not probes:
        return {}

    names = list(probes)
    results = await asyncio.gather(*(_reachable(probes[n]) for n in names))
    return dict(zip(names, results, strict=True))


@router.get("/health/live")
async def live() -> dict[str, str]:
    """Liveness: the process is up. Never touches a dependency."""
    return {"status": "alive"}


@router.get("/health/ready")
async def ready() -> dict[str, Any]:
    """Readiness: can this instance serve a turn?

    A store being down is reported as `degraded`, not as unready, and the
    response stays 200. That is deliberate: this deployment answers from an
    in-process fallback when a backing store disappears, so it can still serve
    -- pulling it out of rotation would trade degraded service for none. The
    defect being fixed was never the 200; it was claiming nothing was wrong.
    """
    try:
        store = get_store()
        documents = int(store.describe().get("documents", 0))
    except RuntimeError:
        return {"status": "not_ready", "reason": "knowledge store uninitialised"}

    if checkpointer_kind() == "uninitialised":
        return {"status": "not_ready", "reason": "graph not compiled"}

    # Initialised is not the same as working. Without this, an instance whose
    # database and cache had both died still answered "ready" and kept taking
    # traffic -- every ticket request stalling until the socket timed out.
    reachable = await _probe_subsystems()
    down = sorted(name for name, ok in reachable.items() if not ok)
    if down:
        return {"status": "degraded", "unreachable": down, "knowledge_documents": documents}

    return {"status": "ready", "knowledge_documents": documents}


@router.get("/health")
async def health() -> dict[str, Any]:
    """Full subsystem report, including which fallbacks are active."""
    settings = get_settings()
    gateway = get_gateway()
    llm = gateway.status()

    try:
        knowledge = dict(get_store().describe())
    except RuntimeError:
        knowledge = {"backend": "uninitialised"}

    try:
        sessions = dict(get_sessions().describe())
    except RuntimeError:
        sessions = {"backend": "uninitialised"}

    degradations: list[str] = []
    if llm["degraded_to_stub"]:
        degradations.append(
            "No LLM provider reachable: replies come from a deterministic offline "
            "stub and are not model-generated."
        )
    if knowledge.get("backend") == "in-memory":
        degradations.append(
            "Qdrant unavailable: hybrid search runs in-process (same dense+BM25+RRF "
            "algorithm, no persistence)."
        )
    if knowledge.get("embedder") == "hashing-fallback":
        degradations.append(
            "sentence-transformers not installed: dense vectors come from a "
            "feature-hashing embedder, not bge-small-en-v1.5."
        )
    if sessions.get("backend") == "in-memory":
        degradations.append("Valkey/Redis unavailable: sessions are per-process and non-durable.")
    if checkpointer_kind() == "in-memory":
        degradations.append(
            "PostgreSQL unavailable: LangGraph checkpoints are in-memory, so "
            "conversations do not survive a restart."
        )
    if not settings.twilio_webhooks_ready:
        degradations.append(
            "Twilio webhooks not configured: inbound SMS and WhatsApp are rejected with "
            "503, because signatures cannot be validated without an auth token and the "
            "public webhook URL."
        )
    if not settings.email_webhooks_ready:
        degradations.append(
            "Email intake not configured: inbound mail is rejected with 503, because a "
            "forged From: header cannot be distinguished without a provider signature."
        )
    from app.services.repository import get_repository

    if get_repository() is None:
        degradations.append(
            "No database configured: tickets live in-process only and are lost on restart."
        )

    # What is actually reachable right now, as opposed to what was wired at boot.
    reachable = await _probe_subsystems()
    unreachable = sorted(name for name, ok in reachable.items() if not ok)
    for name in unreachable:
        degradations.append(
            f"{name} was reachable at startup but is not responding now: requests "
            f"fall back to the in-process store after {PROBE_TIMEOUT_SECONDS:g}s."
        )
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        degradations.append(
            "Langfuse not configured: spans are recorded in-process and inspectable "
            "at /api/traces, but nothing is exported."
        )
    if not settings.livekit_api_key:
        degradations.append("LiveKit not configured: the voice channel is inactive.")

    return {
        "status": "degraded" if degradations else "ok",
        "environment": settings.fastapi_env,
        "subsystems": {
            "llm": llm,
            "knowledge": {**knowledge, "reachable": reachable.get("knowledge")},
            "sessions": {**sessions, "reachable": reachable.get("sessions")},
            "checkpointer": checkpointer_kind(),
            "crm": _crm_report(reachable.get("database")),
            "analytics": get_metrics().describe(),
            "tracing": get_tracer().describe(),
        },
        # null means "not configured, so nothing to probe" -- distinct from false.
        "reachable": reachable,
        "policy": {
            "pii_masking_enabled": settings.pii_masking_enabled,
            "escalation_refund_threshold": settings.escalation_refund_threshold,
            "confidence_escalation_threshold": settings.confidence_escalation_threshold,
            "sentiment_escalation_threshold": settings.sentiment_escalation_threshold,
            "trust_channel_identity": settings.auth_trust_channel_identity,
        },
        "degradations": degradations,
    }
