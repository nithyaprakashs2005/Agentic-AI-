"""Pipecat voice pipeline configuration and worker entry point.

Pipecat is NOT importable in the API's environment: it pins an `openai` major
that conflicts with LiteLLM (see docs/DEVIATIONS.md). So the voice pipeline runs
as its own process from `voice-worker/`, and this module is written so that

* `build_config()` and `preflight()` work with no Pipecat installed -- they are
  pure configuration and are covered by tests;
* `run_pipeline()` imports Pipecat lazily and raises a clear, actionable error
  if it is missing, rather than an ImportError at module load.

That split keeps the configuration honest and testable in CI while the runtime
lives where its dependencies can actually be satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agents.prompts.voice import VOICE_SYSTEM_PROMPT
from app.config import Settings, get_settings
from app.utils.logging import get_logger
from app.voice.latency import LatencyBudget, default_budget

log = get_logger(__name__)


class VoiceDependencyError(RuntimeError):
    """Pipecat and its transports are not installed in this environment."""


@dataclass(slots=True)
class VoicePipelineConfig:
    """Everything needed to construct the Pipecat pipeline."""

    room_url: str
    token: str
    stt_model: str
    llm_model: str
    tts_voice: str
    tts_lang: str
    system_prompt: str
    vad_stop_secs: float = 0.3
    vad_min_volume: float = 0.4
    allow_interruptions: bool = True
    enable_metrics: bool = True
    budget: LatencyBudget = field(default_factory=default_budget)

    def summary(self) -> dict[str, Any]:
        """Loggable, secret-free description."""
        return {
            "room_url": self.room_url,
            "stt_model": self.stt_model,
            "llm_model": self.llm_model,
            "tts_voice": self.tts_voice,
            "vad_stop_secs": self.vad_stop_secs,
            "barge_in": self.allow_interruptions,
            "target_ttfa_ms": self.budget.target_ms,
        }


def build_config(
    *, room_url: str, token: str, settings: Settings | None = None
) -> VoicePipelineConfig:
    """Assemble pipeline configuration from settings. No Pipecat needed."""
    settings = settings or get_settings()
    return VoicePipelineConfig(
        room_url=room_url,
        token=token,
        stt_model=settings.groq_stt_model,
        llm_model=settings.groq_frontier_model,
        tts_voice=settings.kokoro_voice,
        tts_lang=settings.kokoro_lang,
        system_prompt=VOICE_SYSTEM_PROMPT,
        budget=default_budget(settings),
    )


def preflight(settings: Settings | None = None) -> dict[str, Any]:
    """Report whether the voice channel could actually run, and what is missing.

    Called by `/api/voice/status` and by the worker before it starts, so a
    misconfiguration surfaces as a clear message instead of a failed call.
    """
    settings = settings or get_settings()

    missing: list[str] = []
    if not (settings.livekit_api_key and settings.livekit_api_secret):
        missing.append("LIVEKIT_API_KEY / LIVEKIT_API_SECRET")
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY (Whisper STT and LLM)")

    dependencies = {name: _importable(name) for name in ("pipecat", "livekit", "kokoro_onnx")}
    if not dependencies["pipecat"]:
        missing.append("pipecat-ai (install in voice-worker/, not the API env)")

    budget = default_budget(settings)
    return {
        "ready": not missing,
        "missing": missing,
        "dependencies": dependencies,
        "latency_budget": budget.report(),
    }


def _importable(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


async def run_pipeline(config: VoicePipelineConfig) -> None:
    """Build and run the Pipecat pipeline. Requires the voice environment.

    Pipecat's module layout has moved across releases, so imports are resolved
    at call time with an explicit error rather than assumed.
    """
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.services.groq.llm import GroqLLMService
        from pipecat.services.groq.stt import GroqSTTService
        from pipecat.transports.livekit.transport import (
            LiveKitParams,
            LiveKitTransport,
        )
    except ImportError as exc:
        raise VoiceDependencyError(
            "Pipecat is not installed in this environment. The voice pipeline runs "
            "from voice-worker/ with its own virtualenv, because Pipecat and LiteLLM "
            "require incompatible openai versions. See docs/DEVIATIONS.md."
        ) from exc

    settings = get_settings()
    if not settings.groq_api_key:
        raise VoiceDependencyError("GROQ_API_KEY is required for STT and LLM.")

    log.info("voice.pipeline_starting", extra=config.summary())

    transport = LiveKitTransport(
        url=config.room_url,
        token=config.token,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = GroqSTTService(api_key=settings.groq_api_key.get_secret_value(), model=config.stt_model)
    llm = GroqLLMService(api_key=settings.groq_api_key.get_secret_value(), model=config.llm_model)
    tts = _build_tts(config)

    pipeline = Pipeline([transport.input(), stt, llm, tts, transport.output()])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=config.allow_interruptions,
            enable_metrics=config.enable_metrics,
        ),
    )
    await PipelineRunner().run(task)


def _build_tts(config: VoicePipelineConfig) -> Any:
    """Kokoro-82M (Apache 2.0, local) with Edge-TTS as the free fallback."""
    try:
        from pipecat.services.kokoro.tts import KokoroTTSService

        return KokoroTTSService(voice=config.tts_voice, lang=config.tts_lang)
    except ImportError:
        pass

    try:
        from pipecat.services.edge.tts import EdgeTTSService

        log.warning("voice.kokoro_unavailable", extra={"fallback": "edge-tts"})
        return EdgeTTSService(voice=config.tts_voice)
    except ImportError as exc:
        raise VoiceDependencyError(
            "No free TTS backend available. Install kokoro-onnx (preferred, local) "
            "or an Edge-TTS provider in the voice worker environment."
        ) from exc
