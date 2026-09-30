"""Event loop selection shared by every entry point.

psycopg 3 refuses to run async on Windows' `ProactorEventLoop`, which is the
platform default and what `asyncio.run()` gives you. Any entry point that touches
PostgreSQL -- the server, the evaluation harness, future CLI tools -- has to opt
out of it.

This lives in one place because it was originally fixed only in `run.py`, and
`scripts/eval_fcr.py` kept failing to reach the database in exactly the same way.
A shared helper means the next entry point cannot forget.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any


def loop_factory() -> asyncio.AbstractEventLoop:
    """An event loop psycopg can use on every platform.

    Selector loops do not support subprocess transports. Nothing in this service
    spawns subprocesses, so that costs nothing.
    """
    return asyncio.SelectorEventLoop()


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """`asyncio.run` on a psycopg-compatible loop."""
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        return runner.run(coro)
