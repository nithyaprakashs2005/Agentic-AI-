"""Voice channel endpoints: LiveKit room management and pipeline status.

The Pipecat pipeline itself cannot run in this environment -- Pipecat pins an
`openai` major that conflicts with LiteLLM, so it belongs in its own process and
virtualenv (see docs/DEVIATIONS.md). These endpoints therefore report honestly
whether the voice channel is wired, and mint LiveKit tokens when it is, rather
than pretending a pipeline is attached.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.prompts.voice import VOICE_SYSTEM_PROMPT, filler_for
from app.config import get_settings
from app.utils.channel_adapter import to_voice
from app.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])

# A status endpoint nobody waits for is useless, and an unreachable LiveKit is
# the answer we want quickly rather than accurately.
LIVEKIT_PROBE_SECONDS = 1.5


class RoomRequest(BaseModel):
    customer_id: str = Field(default="cust_demo", max_length=64)
    room_name: str | None = None


class VoicePreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


async def _livekit_reachable(url: str) -> bool:
    """Can we actually open a socket to the LiveKit server?

    Credentials being present says nothing about whether the server is up. This
    endpoint previously reported `enabled: true` purely because two environment
    variables were set, while nothing was listening on the port -- the same
    mistake `/health` was making about its backing stores.
    """
    parsed = urlparse(url if "://" in url else f"ws://{url}")
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme in {"wss", "https"} else 80)
    try:
        async with asyncio.timeout(LIVEKIT_PROBE_SECONDS):
            _, writer = await asyncio.open_connection(host, port)
    except Exception:
        return False
    writer.close()
    with suppress(Exception):
        await writer.wait_closed()
    return True


@router.get("/status")
async def voice_status() -> dict[str, Any]:
    """What the voice channel needs, what it has, and what is actually running."""
    settings = get_settings()
    credentials = bool(settings.livekit_api_key and settings.livekit_api_secret)
    reachable = await _livekit_reachable(settings.livekit_url) if credentials else False
    kokoro_present = Path(settings.kokoro_model_path).exists()

    missing: list[str] = []
    if not credentials:
        missing.append("LIVEKIT_API_KEY / LIVEKIT_API_SECRET")
    elif not reachable:
        missing.append(f"a LiveKit server listening on {settings.livekit_url}")
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY (Whisper STT + LLM)")
    if not kokoro_present:
        missing.append(f"the Kokoro model at {settings.kokoro_model_path}")

    # `enabled` means a call could actually be placed, which needs the server up
    # -- not merely that someone set two variables.
    return {
        "enabled": credentials and reachable,
        "credentials_configured": credentials,
        "transport": {
            "url": settings.livekit_url,
            "configured": credentials,
            "reachable": reachable,
        },
        "stt": {"model": settings.groq_stt_model, "configured": bool(settings.groq_api_key)},
        "tts": {
            "voice": settings.kokoro_voice,
            "model_path": settings.kokoro_model_path,
            "model_present": kokoro_present,
        },
        "target_ttfa_ms": settings.target_ttfa_ms,
        "pipeline_process": "external (own virtualenv)",
        "missing_configuration": missing,
    }


@router.post("/rooms")
async def create_room(payload: RoomRequest) -> dict[str, Any]:
    """Mint a LiveKit access token for a browser client."""
    settings = get_settings()
    if not (settings.livekit_api_key and settings.livekit_api_secret):
        raise HTTPException(
            status_code=503,
            detail="Voice channel is not configured on this deployment.",
        )

    try:
        from livekit import api
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Voice dependencies are not installed on this deployment.",
        ) from exc

    room = payload.room_name or f"support-{payload.customer_id}"
    token = (
        api.AccessToken(
            settings.livekit_api_key.get_secret_value(),
            settings.livekit_api_secret.get_secret_value(),
        )
        .with_identity(payload.customer_id)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    log.info("voice.room_token_issued", extra={"room": room})
    return {"room": room, "url": settings.livekit_url, "token": token}


@router.post("/preview")
async def preview_voice_rendering(payload: VoicePreviewRequest) -> dict[str, Any]:
    """Show how text will be spoken after voice adaptation.

    Useful without any voice infrastructure: it exercises the same
    `channel_adapter.to_voice` transform the TTS stage would receive.
    """
    spoken = to_voice(payload.text)
    return {
        "original": payload.text,
        "spoken": spoken,
        "filler_example": filler_for(0),
        "checks": {
            "has_markdown": any(token in spoken for token in ("**", "##", "- ", "```")),
            "has_url": "http" in spoken,
            "has_bare_symbols": any(sym in spoken for sym in ("$", "@", "%", "&")),
        },
        "system_prompt_chars": len(VOICE_SYSTEM_PROMPT),
    }
