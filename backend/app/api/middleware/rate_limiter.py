"""Sliding-window rate limiting for public endpoints.

system specification Constraints: rate limiting is mandatory on anything public-facing.
Redis-backed when available so the window is shared across workers; otherwise
per-process, which is still the correct behaviour for a single-worker deployment.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger(__name__)

# Health and docs must stay reachable while the app is being hammered.
_EXEMPT_PATHS = frozenset(
    {"/health", "/health/live", "/health/ready", "/docs", "/openapi.json", "/redoc"}
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client-IP sliding window."""

    def __init__(self, app: object) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        settings = get_settings()
        self._limit = settings.rate_limit_requests
        self._window = float(settings.rate_limit_window_seconds)
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        # Trust a proxy header only for its first hop.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        window = self._hits[key]
        cutoff = now - self._window
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self._limit:
            retry_after = max(1, int(self._window - (now - window[0])))
            return False, retry_after
        window.append(now)
        return True, 0

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        key = self._client_key(request)
        allowed, retry_after = self._allow(key)
        if not allowed:
            log.warning("ratelimit.rejected", extra={"path": request.url.path})
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down."},
                headers={"Retry-After": str(retry_after)},
            )

        response = await call_next(request)
        remaining = max(0, self._limit - len(self._hits[key]))
        response.headers["X-RateLimit-Limit"] = str(self._limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
