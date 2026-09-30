"""FastAPI application entry point.

Startup wires every subsystem with graceful degradation: a missing backing
service downgrades that subsystem and is reported by `GET /health`, rather than
preventing the process from booting. That is what lets the API be exercised end
to end with no Docker, no API keys, and no database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request

from app.agents.graph import init_graph, shutdown_checkpointer
from app.api.middleware.rate_limiter import RateLimitMiddleware
from app.api.routes import (
    analytics,
    chat,
    email,
    health,
    tickets,
    traces,
    voice,
    webhooks,
)
from app.config import get_settings
from app.ingestion.knowledge_loader import ingest
from app.services.crm import init_crm
from app.services.llm import get_gateway
from app.services.repository import init_repository, shutdown_repository
from app.services.session import init_sessions
from app.services.tracing import init_tracing, shutdown_tracing
from app.services.vector_store import init_store
from app.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


# NOTE: the Windows event loop policy is NOT set here. uvicorn creates its
# loop before importing this module, so setting it here is too late --
# psycopg would still see a ProactorEventLoop. It is set in run.py, which
# is the supported entry point on Windows. See docs/DEVIATIONS.md.


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.litellm_log_level)

    log.info("startup.begin", extra={"environment": settings.fastapi_env})

    # Probe the local model server once so the first customer turn does not pay
    # a connection timeout on a provider that is not there.
    await get_gateway().probe_ollama()

    init_tracing(settings)
    crm = init_crm(settings)

    repository = await init_repository(settings)
    if repository is not None:
        # Mirror the CRM's reference customers so ticket foreign keys resolve.
        try:
            synced = await repository.sync_customers(crm.all_customers())
            log.info("repository.customers_synced", extra={"count": synced})
        except Exception as exc:
            # A database that is slow at boot must not stop the API from
            # serving. Chat degrades to the in-memory store; it does not 503.
            log.warning("repository.customer_sync_failed", extra={"error": str(exc)[:250]})
    await init_sessions(settings)
    store = await init_store(settings)
    await ingest(store)
    await init_graph(settings)

    log.info("startup.complete")
    yield

    await shutdown_tracing()
    await shutdown_repository()
    await shutdown_checkpointer()
    log.info("shutdown.complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Customer Support Voice & Chat AI Agent",
        description=(
            "Multi-channel AI customer support platform. LangGraph orchestrator over "
            "hybrid RAG, with sentiment-driven escalation and CRM ticket lifecycle."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(tickets.router)
    app.include_router(voice.router)
    app.include_router(webhooks.router)
    app.include_router(email.router)
    app.include_router(analytics.router)
    app.include_router(traces.router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Last resort. A customer must never see a stack trace or graph state."""
        log.exception("request.unhandled", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={"detail": "Something went wrong on our side. Please try again."},
        )

    @app.get("/", tags=["meta"])
    async def root() -> dict[str, Any]:
        return {
            "service": "customer-support-ai-agent",
            "docs": "/docs",
            "health": "/health",
            "chat": "/api/chat",
        }

    return app


app = create_app()
