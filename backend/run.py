"""Server entry point.

Exists to control which asyncio event loop the server runs on, which uvicorn
does not let you choose from the CLI.

psycopg 3 refuses to run async on Windows' `ProactorEventLoop`. uvicorn 0.40
picks the loop with a factory that hardcodes it:

    def asyncio_loop_factory(use_subprocess: bool = False):
        if sys.platform == "win32" and not use_subprocess:
            return asyncio.ProactorEventLoop
        return asyncio.SelectorEventLoop

Note it ignores `asyncio.set_event_loop_policy` completely -- setting a policy,
anywhere, has no effect. So this module builds the `uvicorn.Server` itself and
drives it through an `asyncio.Runner` with an explicit `SelectorEventLoop`.

Without this the PostgreSQL checkpointer fails at startup and LangGraph falls
back to in-memory checkpoints. The symptom is quiet: everything works, but
conversations stop surviving a restart, and only on Windows.

    uv run python run.py                 # serve
    uv run python run.py --reload        # reload on changes under app/

On Linux and macOS `SelectorEventLoop` is already the default, so this is a
no-op passthrough.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.utils.runtime import loop_factory


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the support agent API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Reload on changes under app/")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    import uvicorn

    config = uvicorn.Config(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        # Watch only the package: the default watches all of backend/, so test
        # artefacts and __pycache__ churn cause restart storms.
        reload_dirs=["app"] if args.reload else None,
        log_level=args.log_level,
    )

    if args.reload:
        # Reload runs the app in a subprocess, for which uvicorn already
        # selects SelectorEventLoop -- so its own runner is correct here.
        return int(uvicorn.run(**_reload_kwargs(config, args)) or 0)

    server = uvicorn.Server(config)
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        print(  # noqa: T201 -- startup banner, before logging is configured
            f"event loop: {type(runner.get_loop()).__name__}",
            file=sys.stderr,
        )
        runner.run(server.serve())
    return 0


def _reload_kwargs(config: object, args: argparse.Namespace) -> dict[str, object]:
    return {
        "app": "app.main:app",
        "host": args.host,
        "port": args.port,
        "reload": True,
        "reload_dirs": ["app"],
        "log_level": args.log_level,
    }


if __name__ == "__main__":
    raise SystemExit(main())
