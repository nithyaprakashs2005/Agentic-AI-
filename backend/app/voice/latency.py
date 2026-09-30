"""Voice latency budget.

system specification sets a sub-400ms P50 Time-to-First-Audio target, decomposed into
per-stage budgets summing to ~385ms. Encoding that here makes the budget
checkable in CI rather than aspirational prose, and lets a smoke test fail when
a configuration change (a heavier TTS voice, a non-LPU provider) breaks it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings, get_settings


@dataclass(slots=True, frozen=True)
class Stage:
    name: str
    component: str
    budget_ms: int
    concurrent: bool = False
    """True when the stage overlaps an earlier one and does not add to the total."""


# The budget from system specification's latency table. SLM classification runs in parallel
# with STT, so it is tracked but excluded from the serial sum.
DEFAULT_STAGES: tuple[Stage, ...] = (
    Stage("transport_vad", "LiveKit OSS + Silero VAD", 40),
    Stage("stt", "Groq Whisper Large v3", 110),
    Stage("slm_classification", "Groq Llama 3.1 8B", 25, concurrent=True),
    Stage("llm_ttft", "Groq Llama 3.3 70B", 150),
    Stage("tts_first_chunk", "Kokoro-82M (local)", 60),
)


@dataclass(slots=True)
class LatencyBudget:
    target_ms: int
    stages: tuple[Stage, ...] = field(default=DEFAULT_STAGES)

    @property
    def serial_total_ms(self) -> int:
        """Sum of stages on the critical path."""
        return sum(s.budget_ms for s in self.stages if not s.concurrent)

    @property
    def headroom_ms(self) -> int:
        return self.target_ms - self.serial_total_ms

    @property
    def within_target(self) -> bool:
        return self.serial_total_ms <= self.target_ms

    def report(self) -> dict[str, object]:
        return {
            "target_ms": self.target_ms,
            "serial_total_ms": self.serial_total_ms,
            "headroom_ms": self.headroom_ms,
            "within_target": self.within_target,
            "stages": [
                {
                    "name": s.name,
                    "component": s.component,
                    "budget_ms": s.budget_ms,
                    "on_critical_path": not s.concurrent,
                }
                for s in self.stages
            ],
        }


def default_budget(settings: Settings | None = None) -> LatencyBudget:
    settings = settings or get_settings()
    return LatencyBudget(target_ms=settings.target_ttfa_ms)
