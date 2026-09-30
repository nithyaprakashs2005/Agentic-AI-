"""Cross-channel session continuity.

system specification Business Requirements: "a customer who begins on chat and later calls
is never asked to repeat context." The LangGraph checkpointer persists the
conversation itself, keyed by thread id; this store owns the mapping from a
*customer* to their live thread, so any channel can rejoin it.

Valkey/Redis when `REDIS_URL` is set, otherwise an in-process TTL dict.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from app.agents.state import new_conversation_id
from app.config import Settings, get_settings
from app.utils.logging import get_logger

log = get_logger(__name__)

SESSION_TTL_SECONDS = 60 * 60 * 12  # a support thread stays joinable for 12h


@dataclass(slots=True)
class Session:
    conversation_id: str
    customer_id: str
    channels: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turn_count: int = 0
    ticket_id: str | None = None

    def touch(self, channel: str) -> None:
        self.updated_at = time.time()
        self.turn_count += 1
        if channel not in self.channels:
            self.channels.append(channel)

    def dict(self) -> dict[str, Any]:
        return asdict(self)


class SessionBackend(Protocol):
    async def ping(self) -> bool: ...
    async def get(self, conversation_id: str) -> Session | None: ...
    async def put(self, session: Session) -> None: ...
    async def find_by_customer(self, customer_id: str) -> Session | None: ...
    def describe(self) -> dict[str, Any]: ...


class InMemorySessions:
    backend = "in-memory"

    def __init__(self, ttl: int = SESSION_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._by_id: dict[str, Session] = {}
        self._by_customer: dict[str, str] = {}

    async def ping(self) -> bool:
        """In-process: alive whenever the process is."""
        return True

    def _expired(self, session: Session) -> bool:
        return (time.time() - session.updated_at) > self._ttl

    def _sweep(self) -> None:
        stale = [cid for cid, s in self._by_id.items() if self._expired(s)]
        for cid in stale:
            session = self._by_id.pop(cid, None)
            if session and self._by_customer.get(session.customer_id) == cid:
                self._by_customer.pop(session.customer_id, None)

    async def get(self, conversation_id: str) -> Session | None:
        self._sweep()
        return self._by_id.get(conversation_id)

    async def put(self, session: Session) -> None:
        self._by_id[session.conversation_id] = session
        self._by_customer[session.customer_id] = session.conversation_id

    async def find_by_customer(self, customer_id: str) -> Session | None:
        self._sweep()
        conversation_id = self._by_customer.get(customer_id)
        return self._by_id.get(conversation_id) if conversation_id else None

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend, "sessions": len(self._by_id)}


class RedisSessions:
    backend = "redis"

    def __init__(self, client: Any, ttl: int = SESSION_TTL_SECONDS) -> None:
        self._redis = client
        self._ttl = ttl

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"session:{conversation_id}"

    @staticmethod
    def _customer_key(customer_id: str) -> str:
        return f"customer_thread:{customer_id}"

    @classmethod
    async def connect(cls, settings: Settings) -> RedisSessions | None:
        if not settings.has_redis:
            return None
        try:
            import redis.asyncio as redis

            client: Any = redis.from_url(
                str(settings.redis_url), encoding="utf-8", decode_responses=True
            )
            await client.ping()
            log.info("sessions.redis_connected")
            return cls(client)
        except Exception as exc:
            log.warning("sessions.redis_unavailable", extra={"error": str(exc)[:200]})
            return None

    async def ping(self) -> bool:
        """Round-trip to the server, so a vanished peer reads as unreachable."""
        return bool(await self._redis.ping())

    async def get(self, conversation_id: str) -> Session | None:
        raw = await self._redis.get(self._key(conversation_id))
        return Session(**json.loads(raw)) if raw else None

    async def put(self, session: Session) -> None:
        await self._redis.set(
            self._key(session.conversation_id), json.dumps(session.dict()), ex=self._ttl
        )
        await self._redis.set(
            self._customer_key(session.customer_id), session.conversation_id, ex=self._ttl
        )

    async def find_by_customer(self, customer_id: str) -> Session | None:
        conversation_id = await self._redis.get(self._customer_key(customer_id))
        return await self.get(conversation_id) if conversation_id else None

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend}


_sessions: SessionBackend | None = None


async def init_sessions(settings: Settings | None = None) -> SessionBackend:
    global _sessions
    settings = settings or get_settings()
    _sessions = await RedisSessions.connect(settings) or InMemorySessions()
    log.info("sessions.ready", extra=_sessions.describe())
    return _sessions


def get_sessions() -> SessionBackend:
    if _sessions is None:
        raise RuntimeError("session store not initialised; call init_sessions() at startup")
    return _sessions


async def resolve_session(
    *,
    customer_id: str,
    channel: str,
    conversation_id: str | None = None,
) -> Session:
    """Find or open the thread this turn belongs to.

    An explicit `conversation_id` always wins. Otherwise a known customer
    rejoins their live thread -- that is what makes a chat-then-call handover
    keep its context.
    """
    store = get_sessions()

    if conversation_id:
        existing = await store.get(conversation_id)
        if existing:
            existing.touch(channel)
            await store.put(existing)
            return existing
        opened = Session(conversation_id=conversation_id, customer_id=customer_id)
        opened.touch(channel)
        await store.put(opened)
        return opened

    if customer_id:
        joined = await store.find_by_customer(customer_id)
        if joined:
            joined.touch(channel)
            await store.put(joined)
            log.info(
                "session.rejoined",
                extra={
                    "conversation_id": joined.conversation_id,
                    "channels": joined.channels,
                    "turns": joined.turn_count,
                },
            )
            return joined

    fresh = Session(conversation_id=new_conversation_id(), customer_id=customer_id)
    fresh.touch(channel)
    await store.put(fresh)
    return fresh
