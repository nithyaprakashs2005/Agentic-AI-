"""Shared fixtures.

Every test runs fully offline: no API keys, no Docker, no network. The LLM
gateway is forced into stub mode so results are deterministic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from app.config import Settings, get_settings
from app.services import llm as llm_module


@pytest.fixture(autouse=True)
def _offline_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Force a keyless, service-free configuration for every test."""
    get_settings.cache_clear()
    for var in (
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "DATABASE_URL",
        "REDIS_URL",
        "TWILIO_ACCOUNT_SID",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "CRM_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        fastapi_env="development",
    )
    patch_settings(monkeypatch, settings)

    yield settings
    get_settings.cache_clear()


def patch_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """Point every live reference to `get_settings` at `settings`.

    Modules do `from app.config import get_settings`, which binds the function
    *by value* at import time -- so patching `app.config.get_settings` alone
    leaves every importer untouched. A hardcoded module list silently rots as
    modules are added: a module missing from it keeps whatever the previous
    test patched, which leaks configuration across tests. Discovering the
    importers dynamically cannot rot.
    """
    import sys

    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    for name, module in list(sys.modules.items()):
        if not name.startswith("app.") or module is None:
            continue
        if getattr(module, "get_settings", None) is not None:
            monkeypatch.setattr(f"{name}.get_settings", lambda: settings, raising=False)


@pytest.fixture(autouse=True)
def _force_local_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite hermetic regardless of what is running locally.

    Unlike Redis and Postgres, the Qdrant store has no "configured?" setting to
    switch it off -- it always attempts a connection. With qdrant-client
    installed and a container up, the offline fixtures would quietly bind to the
    real server, so results depended on whether Docker happened to be running.
    """

    async def _no_qdrant(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "app.services.vector_store.QdrantHybridStore.connect", _no_qdrant, raising=False
    )

    # Never export spans to a real Langfuse. Like Qdrant, it is attempted
    # whenever keys exist, so with a container running the suite would ship
    # test traffic into a live observability backend.
    monkeypatch.setattr(
        "app.services.tracing.LangfuseTracer.connect",
        classmethod(lambda cls, settings: None),
    )

    # Force the hashing embedder. Loading bge-small per session took the suite
    # from 24s to nine minutes, needs a model download in CI, and makes results
    # depend on which embedder happens to be installed.
    from app.services.vector_store import HashingEmbedder

    monkeypatch.setattr(
        "app.services.vector_store.build_embedder",
        lambda settings: HashingEmbedder(settings.embedding_dim),
    )


@pytest.fixture(autouse=True)
def _stub_gateway(_offline_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway with no reachable provider -> deterministic stub responses."""
    gateway = llm_module.LLMGateway(_offline_settings)
    gateway._ollama_available = False
    monkeypatch.setattr(llm_module, "_gateway", gateway)


@pytest.fixture
async def knowledge_store() -> AsyncIterator[Any]:
    from app.ingestion.knowledge_loader import ingest
    from app.services.vector_store import init_store

    store = await init_store()
    await ingest(store)
    yield store


@pytest.fixture
def crm() -> Any:
    from app.services.crm import init_crm

    return init_crm()


@pytest.fixture
async def compiled_graph(knowledge_store: Any, crm: Any) -> Any:
    from app.agents.graph import init_graph

    return await init_graph()


@pytest.fixture
async def api_client() -> AsyncIterator[Any]:
    """httpx client bound to the ASGI app, running the real lifespan."""
    import httpx
    from asgi_lifespan import LifespanManager

    from app.main import create_app

    app = create_app()
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
