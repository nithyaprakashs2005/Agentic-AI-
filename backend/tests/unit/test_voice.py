"""Voice configuration, latency budget, and dependency preflight.

These run without Pipecat installed, which is the point: the configuration must
be verifiable in the API environment even though the runtime lives elsewhere.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.voice.latency import DEFAULT_STAGES, LatencyBudget, Stage, default_budget
from app.voice.pipeline import (
    VoiceDependencyError,
    build_config,
    preflight,
    run_pipeline,
)


# --- Latency budget -----------------------------------------------------
def test_default_budget_meets_the_claudemd_target() -> None:
    """The stage budgets must actually sum under the stated SLO."""
    budget = LatencyBudget(target_ms=400)
    assert budget.serial_total_ms == 360
    assert budget.within_target is True
    assert budget.headroom_ms == 40


def test_concurrent_stages_are_off_the_critical_path() -> None:
    """SLM classification overlaps STT, so it must not inflate the total."""
    slm = next(s for s in DEFAULT_STAGES if s.name == "slm_classification")
    assert slm.concurrent is True
    with_slm = sum(s.budget_ms for s in DEFAULT_STAGES)
    assert LatencyBudget(target_ms=400).serial_total_ms < with_slm


def test_budget_detects_an_overrun() -> None:
    heavy = LatencyBudget(
        target_ms=400,
        stages=(Stage("stt", "slow cloud STT", 300), Stage("llm", "slow llm", 300)),
    )
    assert heavy.within_target is False
    assert heavy.headroom_ms == -200


def test_budget_target_comes_from_settings() -> None:
    tight = Settings(_env_file=None, target_ttfa_ms=200)  # type: ignore[call-arg]
    assert default_budget(tight).within_target is False


def test_report_shape() -> None:
    report = LatencyBudget(target_ms=400).report()
    assert set(report) == {
        "target_ms",
        "serial_total_ms",
        "headroom_ms",
        "within_target",
        "stages",
    }
    assert len(report["stages"]) == len(DEFAULT_STAGES)  # type: ignore[arg-type]


# --- Configuration ------------------------------------------------------
def test_build_config_reads_every_value_from_settings() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        groq_stt_model="whisper-large-v3",
        groq_frontier_model="groq/llama-3.3-70b-versatile",
        kokoro_voice="af_heart",
        kokoro_lang="en-us",
    )
    config = build_config(room_url="ws://localhost:7880", token="tok", settings=settings)

    assert config.stt_model == "whisper-large-v3"
    assert config.llm_model == "groq/llama-3.3-70b-versatile"
    assert config.tts_voice == "af_heart"
    assert config.allow_interruptions is True  # barge-in required by system specification


def test_config_uses_the_voice_optimised_prompt() -> None:
    config = build_config(room_url="ws://x", token="t")
    assert "NEVER use markdown" in config.system_prompt
    assert "phone call" in config.system_prompt


def test_summary_excludes_the_token() -> None:
    """The summary is logged, so it must not carry the room credential."""
    config = build_config(room_url="ws://x", token="super-secret-token")
    assert "super-secret-token" not in str(config.summary())


def test_vad_stop_secs_matches_the_spec() -> None:
    assert build_config(room_url="ws://x", token="t").vad_stop_secs == 0.3


# --- Preflight ----------------------------------------------------------
def test_preflight_reports_missing_credentials() -> None:
    report = preflight(Settings(_env_file=None))  # type: ignore[call-arg]
    assert report["ready"] is False
    missing = " ".join(report["missing"])  # type: ignore[arg-type]
    assert "LIVEKIT_API_KEY" in missing
    assert "GROQ_API_KEY" in missing


def test_preflight_reports_pipecat_absence_in_this_environment() -> None:
    """Documents the split: Pipecat is deliberately not in the API env."""
    report = preflight()
    assert report["dependencies"]["pipecat"] is False  # type: ignore[index]
    assert any("pipecat-ai" in m for m in report["missing"])  # type: ignore[union-attr]


def test_preflight_includes_the_latency_budget() -> None:
    report = preflight()
    assert report["latency_budget"]["within_target"] is True  # type: ignore[index]


async def test_run_pipeline_fails_with_an_actionable_message() -> None:
    """Without Pipecat, the error must explain the split -- not raise ImportError."""
    config = build_config(room_url="ws://x", token="t")
    with pytest.raises(VoiceDependencyError, match="voice-worker"):
        await run_pipeline(config)
