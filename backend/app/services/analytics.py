"""Operational metrics: FCR, CSAT, escalation rate, and per-node latency.

system specification Phase 7 asks for a dashboard covering FCR, CSAT, sentiment, escalation
rates, and per-node latency. Two definitional choices matter more than the code:

**FCR is measured per conversation, not per turn.** A conversation counts as
first-contact-resolved when it reached `resolved`, was never escalated to a
human, and needed no follow-up contact beyond `FCR_TURN_LIMIT` turns. Counting
per turn would inflate the number massively -- a five-turn conversation that
ends in escalation would still score four "resolved" turns.

**CSAT is only ever computed from real ratings.** There is no sentiment-derived
proxy here. A proxy would produce a plausible number that means nothing, and a
plausible wrong number is worse than an absent one. With no ratings submitted,
CSAT reports `null` with a response count of zero.

Storage is an in-process ring buffer, so metrics reset on restart, exactly like
the other non-durable fallbacks. `GET /health` reports that.
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from app.utils.logging import get_logger

log = get_logger(__name__)

MAX_RECORDS = 10_000
FCR_TURN_LIMIT = 2
"""Turns a conversation may take and still count as first-contact resolved."""

CSAT_SATISFIED_THRESHOLD = 4
"""Ratings at or above this (of 5) count as satisfied, the standard CSAT cut."""


@dataclass(slots=True)
class TurnRecord:
    """One completed agent turn."""

    conversation_id: str
    ticket_id: str | None
    channel: str
    intent: str
    resolution_status: str
    escalated: bool
    escalation_reason: str | None
    sentiment_score: float
    resolution_path: str
    node_timings_ms: dict[str, float]
    total_ms: float
    stubbed: bool
    at: float = field(default_factory=time.time)


@dataclass(slots=True)
class CsatResponse:
    ticket_id: str
    conversation_id: str | None
    score: int
    comment: str = ""
    at: float = field(default_factory=time.time)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Returns 0.0 for an empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(fraction * len(ordered))))
    return round(ordered[rank - 1], 2)


class MetricsStore:
    """In-process metrics over a bounded window of recent turns."""

    backend = "in-memory"

    def __init__(self, max_records: int = MAX_RECORDS) -> None:
        self._turns: deque[TurnRecord] = deque(maxlen=max_records)
        self._csat: deque[CsatResponse] = deque(maxlen=max_records)

    # --- ingestion ---
    def record_turn(self, record: TurnRecord) -> None:
        self._turns.append(record)

    def record_csat(self, response: CsatResponse) -> None:
        self._csat.append(response)
        log.info(
            "analytics.csat_recorded",
            extra={"ticket_id": response.ticket_id, "score": response.score},
        )

    # --- queries ---
    def _window(self, since: float | None) -> list[TurnRecord]:
        if since is None:
            return list(self._turns)
        return [t for t in self._turns if t.at >= since]

    def snapshot(self, window_seconds: float | None = None) -> dict[str, Any]:
        since = time.time() - window_seconds if window_seconds else None
        turns = self._window(since)
        csat = [c for c in self._csat if since is None or c.at >= since]

        if not turns:
            return _empty_snapshot(window_seconds)

        conversations = _group_conversations(turns)

        return {
            "window_seconds": window_seconds,
            "generated_at": time.time(),
            "totals": {
                "turns": len(turns),
                "conversations": len(conversations),
                "tickets": len({t.ticket_id for t in turns if t.ticket_id}),
            },
            "fcr": _fcr(conversations),
            "escalation": _escalation(conversations, turns),
            "csat": _csat(csat, len(conversations)),
            "sentiment": _sentiment(turns),
            "latency": _latency(turns),
            "distribution": {
                "by_intent": dict(Counter(t.intent for t in turns).most_common()),
                "by_channel": dict(Counter(t.channel for t in turns).most_common()),
                "by_resolution_path": dict(Counter(t.resolution_path for t in turns)),
            },
            "data_quality": _data_quality(turns),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "turns_recorded": len(self._turns),
            "csat_responses": len(self._csat),
            "capacity": self._turns.maxlen,
        }

    def reset(self) -> None:
        self._turns.clear()
        self._csat.clear()


# --- Aggregation helpers ------------------------------------------------
def _group_conversations(turns: list[TurnRecord]) -> dict[str, list[TurnRecord]]:
    grouped: dict[str, list[TurnRecord]] = {}
    for turn in turns:
        grouped.setdefault(turn.conversation_id, []).append(turn)
    return grouped


def _fcr(conversations: dict[str, list[TurnRecord]]) -> dict[str, Any]:
    """First-contact resolution, measured per conversation."""
    resolved_first_contact = 0
    for turns in conversations.values():
        escalated_ever = any(t.escalated for t in turns)
        final = turns[-1]
        if (
            not escalated_ever
            and final.resolution_status == "resolved"
            and len(turns) <= FCR_TURN_LIMIT
        ):
            resolved_first_contact += 1

    total = len(conversations)
    return {
        "rate": round(resolved_first_contact / total, 4) if total else None,
        "resolved_first_contact": resolved_first_contact,
        "conversations": total,
        "target": 0.92,
        "definition": (f"resolved, never escalated, and completed within {FCR_TURN_LIMIT} turns"),
    }


def _escalation(
    conversations: dict[str, list[TurnRecord]], turns: list[TurnRecord]
) -> dict[str, Any]:
    escalated = sum(1 for ts in conversations.values() if any(t.escalated for t in ts))
    reasons = Counter(t.escalation_reason for t in turns if t.escalated and t.escalation_reason)
    total = len(conversations)
    return {
        "rate": round(escalated / total, 4) if total else None,
        "escalated_conversations": escalated,
        "conversations": total,
        "by_reason": dict(reasons.most_common()),
    }


def _csat(responses: list[CsatResponse], conversations: int) -> dict[str, Any]:
    """CSAT from submitted ratings only. Never inferred."""
    if not responses:
        return {
            "score": None,
            "responses": 0,
            "response_rate": 0.0,
            "note": "no ratings submitted; CSAT is never inferred from sentiment",
        }

    scores = [r.score for r in responses]
    satisfied = sum(1 for s in scores if s >= CSAT_SATISFIED_THRESHOLD)
    return {
        "score": round(satisfied / len(scores), 4),
        "mean_rating": round(sum(scores) / len(scores), 2),
        "responses": len(responses),
        "response_rate": round(len(responses) / conversations, 4) if conversations else 0.0,
        "distribution": {str(k): v for k, v in sorted(Counter(scores).items())},
        "note": f"satisfied = rating >= {CSAT_SATISFIED_THRESHOLD} of 5",
    }


def _sentiment(turns: list[TurnRecord]) -> dict[str, Any]:
    scores = [t.sentiment_score for t in turns]
    negative = sum(1 for s in scores if s < -0.3)
    positive = sum(1 for s in scores if s > 0.3)
    return {
        "mean": round(sum(scores) / len(scores), 3),
        "p10": percentile(scores, 0.10),
        "p90": percentile(scores, 0.90),
        "negative_share": round(negative / len(scores), 4),
        "positive_share": round(positive / len(scores), 4),
    }


def _latency(turns: list[TurnRecord]) -> dict[str, Any]:
    totals = [t.total_ms for t in turns]

    per_node: dict[str, list[float]] = {}
    for turn in turns:
        for node, ms in turn.node_timings_ms.items():
            per_node.setdefault(node, []).append(ms)

    nodes = {
        node: {
            "p50": percentile(samples, 0.50),
            "p95": percentile(samples, 0.95),
            "max": round(max(samples), 2),
            "samples": len(samples),
        }
        for node, samples in sorted(per_node.items(), key=lambda kv: -percentile(kv[1], 0.50))
    }

    return {
        "turn_total": {
            "p50": percentile(totals, 0.50),
            "p95": percentile(totals, 0.95),
            "max": round(max(totals), 2) if totals else 0.0,
        },
        "by_node": nodes,
    }


def _data_quality(turns: list[TurnRecord]) -> dict[str, Any]:
    """How much of this sample came from the offline stub.

    Metrics computed over stubbed turns describe the stub's keyword heuristics,
    not model behaviour. Reporting the share stops the dashboard being read as
    production quality when it is not.
    """
    stubbed = sum(1 for t in turns if t.stubbed)
    share = round(stubbed / len(turns), 4)
    return {
        "stubbed_share": share,
        "stubbed_turns": stubbed,
        "warning": (
            "Some or all turns ran on the offline stub; these figures reflect "
            "keyword heuristics, not model behaviour."
        )
        if share > 0
        else None,
    }


def _empty_snapshot(window_seconds: float | None) -> dict[str, Any]:
    return {
        "window_seconds": window_seconds,
        "generated_at": time.time(),
        "totals": {"turns": 0, "conversations": 0, "tickets": 0},
        "fcr": {
            "rate": None,
            "resolved_first_contact": 0,
            "conversations": 0,
            "target": 0.92,
            "definition": (
                f"resolved, never escalated, and completed within {FCR_TURN_LIMIT} turns"
            ),
        },
        "escalation": {
            "rate": None,
            "escalated_conversations": 0,
            "conversations": 0,
            "by_reason": {},
        },
        "csat": {
            "score": None,
            "responses": 0,
            "response_rate": 0.0,
            "note": "no ratings submitted; CSAT is never inferred from sentiment",
        },
        "sentiment": {},
        "latency": {"turn_total": {}, "by_node": {}},
        "distribution": {"by_intent": {}, "by_channel": {}, "by_resolution_path": {}},
        "data_quality": {"stubbed_share": 0.0, "stubbed_turns": 0, "warning": None},
    }


_metrics: MetricsStore | None = None


def get_metrics() -> MetricsStore:
    global _metrics
    if _metrics is None:
        _metrics = MetricsStore()
    return _metrics


def record_turn_from_state(
    *,
    conversation_id: str,
    state: dict[str, Any],
    total_ms: float,
) -> None:
    """Convenience hook for the turn path. Never raises into the request."""
    try:
        get_metrics().record_turn(
            TurnRecord(
                conversation_id=conversation_id,
                ticket_id=state.get("ticket_id"),
                channel=str(state.get("channel", "unknown")),
                intent=str(state.get("intent", "general")),
                resolution_status=str(state.get("resolution_status", "pending")),
                escalated=bool(state.get("escalation_required")),
                escalation_reason=state.get("escalation_reason"),
                sentiment_score=float(state.get("sentiment_score", 0.0)),
                resolution_path=str(state.get("resolution_path", "simple")),
                node_timings_ms=dict(state.get("node_timings_ms", {})),
                total_ms=total_ms,
                stubbed=bool(state.get("llm_stubbed")),
            )
        )
    except Exception:
        log.warning("analytics.record_failed", exc_info=True)
