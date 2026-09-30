"""Analytics aggregation, FCR/CSAT definitions, and the metrics endpoints."""

from __future__ import annotations

import time
from typing import Any

import pytest

from app.services.analytics import (
    CSAT_SATISFIED_THRESHOLD,
    FCR_TURN_LIMIT,
    CsatResponse,
    MetricsStore,
    TurnRecord,
    percentile,
)

pytestmark = pytest.mark.timeout(90)


def turn(
    conversation_id: str = "c1",
    *,
    escalated: bool = False,
    status: str = "resolved",
    reason: str | None = None,
    sentiment: float = 0.0,
    intent: str = "billing",
    channel: str = "webchat",
    timings: dict[str, float] | None = None,
    total_ms: float = 100.0,
    stubbed: bool = False,
    at: float | None = None,
) -> TurnRecord:
    record = TurnRecord(
        conversation_id=conversation_id,
        ticket_id=f"TKT-{conversation_id}",
        channel=channel,
        intent=intent,
        resolution_status=status,
        escalated=escalated,
        escalation_reason=reason,
        sentiment_score=sentiment,
        resolution_path="simple",
        node_timings_ms=timings or {"router": 5.0, "agentic_rag": 20.0},
        total_ms=total_ms,
        stubbed=stubbed,
    )
    if at is not None:
        record.at = at
    return record


@pytest.fixture
def store() -> MetricsStore:
    return MetricsStore()


# --- Percentiles --------------------------------------------------------
def test_percentile_of_empty_sample_is_zero() -> None:
    assert percentile([], 0.5) == 0.0


def test_percentile_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert percentile(values, 0.50) == 5.0
    assert percentile(values, 0.95) == 10.0
    assert percentile(values, 0.0) == 1.0


def test_percentile_of_single_value() -> None:
    assert percentile([42.0], 0.95) == 42.0


# --- FCR ----------------------------------------------------------------
def test_fcr_counts_a_clean_single_turn_conversation(store: MetricsStore) -> None:
    store.record_turn(turn("c1"))
    fcr = store.snapshot()["fcr"]
    assert fcr["rate"] == 1.0
    assert fcr["resolved_first_contact"] == 1


def test_fcr_excludes_escalated_conversations(store: MetricsStore) -> None:
    store.record_turn(turn("c1", escalated=True, status="escalated"))
    assert store.snapshot()["fcr"]["rate"] == 0.0


def test_fcr_excludes_a_conversation_that_escalated_at_any_point(
    store: MetricsStore,
) -> None:
    """An early escalation must not be forgiven by a later resolved turn."""
    store.record_turn(turn("c1", escalated=True, status="escalated"))
    store.record_turn(turn("c1", status="resolved"))
    assert store.snapshot()["fcr"]["rate"] == 0.0


def test_fcr_excludes_conversations_over_the_turn_limit(store: MetricsStore) -> None:
    for _ in range(FCR_TURN_LIMIT + 1):
        store.record_turn(turn("c1"))
    assert store.snapshot()["fcr"]["rate"] == 0.0


def test_fcr_allows_up_to_the_turn_limit(store: MetricsStore) -> None:
    for _ in range(FCR_TURN_LIMIT):
        store.record_turn(turn("c1"))
    assert store.snapshot()["fcr"]["rate"] == 1.0


def test_fcr_is_per_conversation_not_per_turn(store: MetricsStore) -> None:
    """Per-turn counting would inflate this to 0.75 instead of 0.5."""
    store.record_turn(turn("clean"))
    store.record_turn(turn("messy", status="resolved"))
    store.record_turn(turn("messy", status="resolved"))
    store.record_turn(turn("messy", escalated=True, status="escalated"))

    snapshot = store.snapshot()
    assert snapshot["totals"]["conversations"] == 2
    assert snapshot["fcr"]["rate"] == 0.5


def test_fcr_reports_the_target_and_its_definition(store: MetricsStore) -> None:
    store.record_turn(turn())
    fcr = store.snapshot()["fcr"]
    assert fcr["target"] == 0.92
    assert "never escalated" in fcr["definition"]


# --- Escalation ---------------------------------------------------------
def test_escalation_rate_and_reasons(store: MetricsStore) -> None:
    store.record_turn(turn("c1", escalated=True, reason="explicit_keyword:human"))
    store.record_turn(turn("c2", escalated=True, reason="explicit_keyword:human"))
    store.record_turn(turn("c3", escalated=True, reason="low_intent_confidence"))
    store.record_turn(turn("c4"))

    escalation = store.snapshot()["escalation"]
    assert escalation["rate"] == 0.75
    assert escalation["by_reason"]["explicit_keyword:human"] == 2
    assert escalation["by_reason"]["low_intent_confidence"] == 1


# --- CSAT ---------------------------------------------------------------
def test_csat_is_null_with_no_ratings(store: MetricsStore) -> None:
    """CSAT must never be inferred from sentiment."""
    store.record_turn(turn(sentiment=0.9))
    csat = store.snapshot()["csat"]
    assert csat["score"] is None
    assert csat["responses"] == 0
    assert "never inferred" in csat["note"]


def test_csat_uses_the_standard_satisfied_cut(store: MetricsStore) -> None:
    store.record_turn(turn("c1"))
    store.record_turn(turn("c2"))
    store.record_turn(turn("c3"))
    store.record_turn(turn("c4"))
    for score in (5, 4, 3, 1):
        store.record_csat(CsatResponse(ticket_id="t", conversation_id=None, score=score))

    csat = store.snapshot()["csat"]
    assert CSAT_SATISFIED_THRESHOLD == 4
    assert csat["score"] == 0.5  # 5 and 4 are satisfied; 3 and 1 are not
    assert csat["mean_rating"] == 3.25
    assert csat["responses"] == 4


def test_csat_response_rate_is_relative_to_conversations(store: MetricsStore) -> None:
    for i in range(4):
        store.record_turn(turn(f"c{i}"))
    store.record_csat(CsatResponse(ticket_id="t", conversation_id=None, score=5))
    assert store.snapshot()["csat"]["response_rate"] == 0.25


def test_csat_distribution(store: MetricsStore) -> None:
    store.record_turn(turn())
    for score in (5, 5, 3):
        store.record_csat(CsatResponse(ticket_id="t", conversation_id=None, score=score))
    assert store.snapshot()["csat"]["distribution"] == {"3": 1, "5": 2}


# --- Latency ------------------------------------------------------------
def test_latency_percentiles_per_node(store: MetricsStore) -> None:
    for ms in (10.0, 20.0, 30.0, 40.0, 100.0):
        store.record_turn(turn(f"c{ms}", timings={"router": ms}, total_ms=ms * 2))

    latency = store.snapshot()["latency"]
    assert latency["by_node"]["router"]["p50"] == 30.0
    assert latency["by_node"]["router"]["max"] == 100.0
    assert latency["by_node"]["router"]["samples"] == 5
    assert latency["turn_total"]["max"] == 200.0


def test_nodes_are_ordered_slowest_first(store: MetricsStore) -> None:
    """The dashboard should lead with the bottleneck."""
    store.record_turn(turn(timings={"fast": 1.0, "slow": 500.0, "middling": 50.0}))
    assert list(store.snapshot()["latency"]["by_node"]) == ["slow", "middling", "fast"]


# --- Sentiment ----------------------------------------------------------
def test_sentiment_shares(store: MetricsStore) -> None:
    store.record_turn(turn("c1", sentiment=-0.8))
    store.record_turn(turn("c2", sentiment=-0.5))
    store.record_turn(turn("c3", sentiment=0.0))
    store.record_turn(turn("c4", sentiment=0.9))

    sentiment = store.snapshot()["sentiment"]
    assert sentiment["negative_share"] == 0.5
    assert sentiment["positive_share"] == 0.25
    assert sentiment["mean"] == pytest.approx(-0.1, abs=0.01)


# --- Data quality -------------------------------------------------------
def test_stubbed_share_is_reported_with_a_warning(store: MetricsStore) -> None:
    """Metrics over stubbed turns describe heuristics, not model behaviour."""
    store.record_turn(turn("c1", stubbed=True))
    store.record_turn(turn("c2", stubbed=False))

    quality = store.snapshot()["data_quality"]
    assert quality["stubbed_share"] == 0.5
    assert quality["warning"] is not None


def test_no_warning_when_nothing_was_stubbed(store: MetricsStore) -> None:
    store.record_turn(turn(stubbed=False))
    assert store.snapshot()["data_quality"]["warning"] is None


# --- Windows and bounds -------------------------------------------------
def test_window_excludes_older_records(store: MetricsStore) -> None:
    now = time.time()
    store.record_turn(turn("old", at=now - 7200))
    store.record_turn(turn("recent", at=now - 60))

    assert store.snapshot(3600)["totals"]["conversations"] == 1
    assert store.snapshot()["totals"]["conversations"] == 2


def test_empty_snapshot_is_well_formed(store: MetricsStore) -> None:
    snapshot = store.snapshot()
    assert snapshot["totals"]["turns"] == 0
    assert snapshot["fcr"]["rate"] is None
    assert snapshot["csat"]["score"] is None
    assert snapshot["escalation"]["rate"] is None


def test_ring_buffer_bounds_memory() -> None:
    small = MetricsStore(max_records=5)
    for i in range(20):
        small.record_turn(turn(f"c{i}"))
    assert small.describe()["turns_recorded"] == 5


def test_distribution_counts(store: MetricsStore) -> None:
    store.record_turn(turn("c1", intent="billing", channel="sms"))
    store.record_turn(turn("c2", intent="billing", channel="voice"))
    store.record_turn(turn("c3", intent="returns", channel="sms"))

    distribution = store.snapshot()["distribution"]
    assert distribution["by_intent"] == {"billing": 2, "returns": 1}
    assert distribution["by_channel"]["sms"] == 2


# --- Endpoints ----------------------------------------------------------
async def test_summary_reflects_real_traffic(api_client: Any) -> None:
    from app.services.analytics import get_metrics

    get_metrics().reset()

    await api_client.post(
        "/api/chat",
        json={"message": "How long do refunds take?", "conversation_id": "an-1"},
    )
    await api_client.post(
        "/api/chat",
        json={"message": "I want to speak to a human", "conversation_id": "an-2"},
    )

    body = (await api_client.get("/api/analytics/summary?window=1h")).json()
    assert body["totals"]["conversations"] == 2
    assert body["escalation"]["escalated_conversations"] == 1
    assert body["latency"]["by_node"], "per-node latency should be captured"
    assert body["data_quality"]["stubbed_share"] == 1.0  # offline in tests


async def test_summary_rejects_an_unknown_window(api_client: Any) -> None:
    assert (await api_client.get("/api/analytics/summary?window=99y")).status_code == 422


async def test_feedback_round_trip(api_client: Any) -> None:
    from app.services.analytics import get_metrics

    get_metrics().reset()
    chat = await api_client.post(
        "/api/chat", json={"message": "How long do refunds take?", "conversation_id": "an-fb"}
    )
    ticket_id = chat.json()["ticket_id"]

    resp = await api_client.post(
        "/api/analytics/feedback", json={"ticket_id": ticket_id, "score": 5}
    )
    assert resp.status_code == 200

    csat = (await api_client.get("/api/analytics/summary?window=1h")).json()["csat"]
    assert csat["responses"] == 1
    assert csat["score"] == 1.0


async def test_feedback_on_an_unknown_ticket_is_404(api_client: Any) -> None:
    resp = await api_client.post(
        "/api/analytics/feedback", json={"ticket_id": "TKT-nope", "score": 5}
    )
    assert resp.status_code == 404


@pytest.mark.parametrize("score", [0, 6, -1, 100])
async def test_feedback_rejects_out_of_range_scores(score: int, api_client: Any) -> None:
    resp = await api_client.post(
        "/api/analytics/feedback", json={"ticket_id": "TKT-x", "score": score}
    )
    assert resp.status_code == 422


async def test_feedback_comment_is_pii_scrubbed(api_client: Any) -> None:
    from app.services.analytics import get_metrics

    get_metrics().reset()
    chat = await api_client.post(
        "/api/chat", json={"message": "How long do refunds take?", "conversation_id": "an-pii"}
    )
    ticket_id = chat.json()["ticket_id"]

    await api_client.post(
        "/api/analytics/feedback",
        json={
            "ticket_id": ticket_id,
            "score": 4,
            "comment": "call me on 555-123-4567 or card 4111 1111 1111 1111",
        },
    )
    stored = get_metrics()._csat[-1]
    assert "4111 1111 1111 1111" not in stored.comment
    assert "555-123-4567" not in stored.comment


async def test_analytics_health_declares_non_durability(api_client: Any) -> None:
    body = (await api_client.get("/api/analytics/health")).json()
    assert body["durable"] is False
    assert body["backend"] == "in-memory"
