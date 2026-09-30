"""The evaluation harness itself: case loading, scoring, and the real suite.

The harness is what the 92% claim will rest on, so its arithmetic is tested
independently of the system it measures.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.evals.harness import CaseResult, EvalCase, load_cases, run_suite, score

pytestmark = pytest.mark.timeout(180)


def case(
    case_id: str = "c1",
    *,
    intent: str = "billing",
    escalate: bool = False,
    sources: list[str] | None = None,
    reason: str | None = None,
) -> EvalCase:
    return EvalCase(
        id=case_id,
        message="test",
        expected_intent=intent,
        should_escalate=escalate,
        expected_sources=sources or [],
        expected_escalation_reason=reason,
    )


def result(
    eval_case: EvalCase,
    *,
    intent: str | None = None,
    escalated: bool | None = None,
    status: str = "resolved",
    citations: list[str] | None = None,
    reason: str | None = None,
    stubbed: bool = False,
    error: str | None = None,
) -> CaseResult:
    return CaseResult(
        case=eval_case,
        intent=intent if intent is not None else eval_case.expected_intent,
        escalated=escalated if escalated is not None else eval_case.should_escalate,
        escalation_reason=reason,
        resolution_status=status,
        citations=citations or [],
        stubbed=stubbed,
        latency_ms=10.0,
        error=error,
    )


# --- Case file ----------------------------------------------------------
def test_the_shipped_case_file_loads() -> None:
    cases = load_cases()
    assert len(cases) >= 30


def test_case_ids_are_unique() -> None:
    ids = [c.id for c in load_cases()]
    assert len(ids) == len(set(ids))


def test_cases_cover_both_outcomes_and_every_intent() -> None:
    """A set that is all one outcome cannot measure precision and recall."""
    cases = load_cases()
    assert sum(1 for c in cases if c.should_escalate) >= 5
    assert sum(1 for c in cases if not c.should_escalate) >= 10

    from app.agents.state import INTENTS

    covered = {c.expected_intent for c in cases}
    assert covered == set(INTENTS), f"intents not covered: {set(INTENTS) - covered}"


def test_expected_sources_reference_real_articles() -> None:
    """A label pointing at a nonexistent article would never be satisfiable."""
    from app.ingestion.knowledge_loader import load_articles

    known = {a["id"] for a in load_articles()}
    for eval_case in load_cases():
        for source in eval_case.expected_sources:
            assert source in known, f"{eval_case.id} expects unknown article {source}"


# --- Scoring ------------------------------------------------------------
def test_fcr_counts_only_cases_needing_no_human() -> None:
    resolve_a, resolve_b = case("a"), case("b")
    needs_human = case("c", escalate=True)

    report = score(
        [
            result(resolve_a),
            result(resolve_b, escalated=True, status="escalated"),  # wrongly escalated
            result(needs_human, status="escalated"),
        ]
    )
    assert report["fcr"]["eligible"] == 2
    assert report["fcr"]["deflected"] == 1
    assert report["fcr"]["rate"] == 0.5


def test_fcr_meets_target_flag() -> None:
    cases = [case(f"c{i}") for i in range(10)]
    report = score([result(c) for c in cases])
    assert report["fcr"]["rate"] == 1.0
    assert report["fcr"]["meets_target"] is True


def test_confusion_matrix_counts_all_four_cells() -> None:
    report = score(
        [
            result(case("tp", escalate=True)),  # should escalate, did
            result(case("fn", escalate=True), escalated=False),  # should, did not
            result(case("fp"), escalated=True),  # should not, did
            result(case("tn")),  # should not, did not
        ]
    )
    confusion = report["escalation"]["confusion"]
    assert confusion == {
        "true_positive": 1,
        "false_negative": 1,
        "false_positive": 1,
        "true_negative": 1,
    }
    assert report["escalation"]["precision"] == 0.5
    assert report["escalation"]["recall"] == 0.5


def test_false_negative_is_called_out_as_the_serious_direction() -> None:
    report = score([result(case("fn", escalate=True), escalated=False)])
    assert "false_negative is the serious" in report["escalation"]["note"]


def test_escalating_everything_scores_poor_precision() -> None:
    """A system that escalates everything must not look good."""
    cases = [case(f"r{i}") for i in range(9)] + [case("e", escalate=True)]
    report = score([result(c, escalated=True, status="escalated") for c in cases])
    assert report["escalation"]["recall"] == 1.0
    assert report["escalation"]["precision"] == 0.1
    assert report["fcr"]["rate"] == 0.0


def test_escalating_nothing_scores_zero_recall() -> None:
    cases = [case(f"e{i}", escalate=True) for i in range(5)]
    report = score([result(c, escalated=False) for c in cases])
    assert report["escalation"]["recall"] == 0.0


def test_intent_accuracy() -> None:
    report = score(
        [
            result(case("a", intent="billing")),
            result(case("b", intent="returns"), intent="billing"),
        ]
    )
    assert report["intent_accuracy"] == 0.5


def test_grounding_matches_on_the_article_prefix() -> None:
    """Citations are chunk ids; a chunk of the right article counts."""
    report = score(
        [result(case("a", sources=["kb-refund-timing"]), citations=["kb-refund-timing#c1"])]
    )
    assert report["citation_grounding"] == 1.0


def test_grounding_ignores_cases_with_no_expected_sources() -> None:
    report = score([result(case("a")), result(case("b", sources=["kb-x"]), citations=["kb-x"])])
    assert report["citation_grounding"] == 1.0  # only the labelled case counts


def test_wrong_source_is_not_grounded() -> None:
    report = score(
        [result(case("a", sources=["kb-refund-timing"]), citations=["kb-return-window"])]
    )
    assert report["citation_grounding"] == 0.0


def test_reason_accuracy_uses_prefix_matching() -> None:
    report = score(
        [
            result(
                case("a", escalate=True, reason="explicit_keyword"),
                reason="explicit_keyword:human",
            )
        ]
    )
    assert report["escalation"]["reason_accuracy"] == 1.0


def test_errored_cases_are_separated_not_silently_passed() -> None:
    report = score([result(case("ok")), result(case("bad"), error="boom")])
    assert report["cases"]["scored"] == 1
    assert report["cases"]["errored"] == 1
    assert report["errors"][0]["id"] == "bad"


def test_failures_list_explains_each_miss() -> None:
    report = score([result(case("a", intent="billing"), intent="returns")])
    assert len(report["failures"]) == 1
    assert report["failures"][0]["got_intent"] == "returns"


def test_stubbed_share_is_reported_with_a_warning() -> None:
    report = score([result(case("a"), stubbed=True)])
    assert report["data_quality"]["stubbed_share"] == 1.0
    assert "keyword heuristics" in report["data_quality"]["warning"]


def test_no_warning_when_a_real_model_was_used() -> None:
    report = score([result(case("a"), stubbed=False)])
    assert report["data_quality"]["warning"] is None


def test_empty_run() -> None:
    assert "error" in score([])


# --- The real suite -----------------------------------------------------
async def test_full_suite_runs_and_reports(compiled_graph: Any) -> None:
    """Runs every labelled case through the real graph.

    Asserts the harness produces a coherent report, not that the system hits a
    particular score -- the score is the finding, and pinning it here would make
    this test a ratchet that blocks honest regressions from surfacing.
    """
    cases = load_cases()
    results = await run_suite(compiled_graph, cases)
    report = score(results)

    assert report["cases"]["errored"] == 0, report["errors"]
    assert report["cases"]["scored"] == len(cases)
    assert 0.0 <= (report["fcr"]["rate"] or 0.0) <= 1.0
    assert report["intent_accuracy"] is not None
    # Offline, the harness must say so.
    assert report["data_quality"]["stubbed_share"] == 1.0
    assert report["data_quality"]["warning"] is not None


async def test_explicit_human_requests_are_never_missed(compiled_graph: Any) -> None:
    """The one failure mode with no acceptable rate: asked for a human, denied one."""
    cases = [c for c in load_cases() if c.expected_escalation_reason == "explicit_keyword"]
    assert cases, "expected explicit-request cases in the set"

    results = await run_suite(compiled_graph, cases)
    missed = [r.case.id for r in results if not r.escalated]
    assert not missed, f"explicit requests for a human were not escalated: {missed}"
