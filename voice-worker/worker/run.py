"""Voice worker entry point.

Joins a LiveKit room and runs the Pipecat pipeline for one call.

    uv run python -m worker.run --room support-demo

Reuses the backend's configuration and prompts by importing from `app.*`, so the
voice channel cannot drift from the text channel on prompts or thresholds. That
requires the backend on PYTHONPATH:

    PYTHONPATH=../backend uv run python -m worker.run --room support-demo

The backend is imported for *configuration only* -- no LiteLLM call is made from
this process, which is what keeps the two dependency trees apart.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Make the backend package importable for shared config and prompts.
_BACKEND = Path(__file__).resolve().parents[2] / "backend"
if _BACKEND.exists() and str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

log = logging.getLogger("voice-worker")


def _mint_token(room: str, identity: str) -> str:
    """Create a LiveKit access token for this worker."""
    from app.config import get_settings
    from livekit import api

    settings = get_settings()
    if not (settings.livekit_api_key and settings.livekit_api_secret):
        raise SystemExit(
            "LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required. "
            "Set them in the repo-root .env."
        )

    return (
        api.AccessToken(
            settings.livekit_api_key.get_secret_value(),
            settings.livekit_api_secret.get_secret_value(),
        )
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Pipecat voice pipeline.")
    parser.add_argument("--room", default="support-demo", help="LiveKit room to join")
    parser.add_argument("--identity", default="support-agent", help="Worker identity")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report readiness and the latency budget, then exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        from app.config import get_settings
        from app.voice.pipeline import VoiceDependencyError, build_config, preflight, run_pipeline
    except ImportError as exc:
        print(
            "Could not import the backend package. Run with the backend on PYTHONPATH:\n"
            "  PYTHONPATH=../backend uv run python -m worker.run --room support-demo",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    report = preflight()
    print(json.dumps(report, indent=2))

    if args.check:
        return 0 if report["ready"] else 1

    if not report["ready"]:
        print(
            "\nNot ready to start. Missing: " + ", ".join(report["missing"]),
            file=sys.stderr,
        )
        return 1

    settings = get_settings()
    config = build_config(
        room_url=settings.livekit_url,
        token=_mint_token(args.room, args.identity),
    )

    log.info("joining room %s", args.room)
    try:
        await run_pipeline(config)
    except VoiceDependencyError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        log.info("interrupted; shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
