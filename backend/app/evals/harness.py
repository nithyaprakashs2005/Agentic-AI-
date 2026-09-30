"""Evaluation harness for the 92% FCR target.

system specification sets 92% first-contact resolution as the headline business metric but
provides no way to measure it. This runs the real compiled graph over a labelled
set and reports where the system actually lands.

Four things are scored, because "FCR" alone hides how it was achieved:

* **Deflection** -- of cases a human should NOT be needed for, how many resolved
  without one. This is the FCR number.
* **Escalation precision and recall** -- a system that escalates everything gets
  perfect "no wrong answers" and is useless; one that escalates nothing traps
  angry customers. Both failure directions are counted separately, and a
  false negative (should have escalated, did not) is the more serious one.
* **Intent accuracy** -- misrouting sends retrieval to the wrong KB category.
* **Citation grounding** -- did the answer draw on the article a correct answer
  needs? An unsourced right-sounding answer is not a resolution.

The report always carries `stubbed_share`. A run against the offline stub scores
keyword heuristics, not a model, and the harness says so rather than letting the
number be quoted as a system capability.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from app.agents.state import initial_state
from app.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "evals" / "fcr_cases.json"


@dataclass(slots=True)
class EvalCase:
    id: str
    message: str
    expected_intent: str
    should_escalate: bool
    expected_sources: list[str] = field(default_factory=list)
    expected_escalation_reason: str | None = None
    customer_id: str = "cust_demo"
    customer_tier: str = "standard"
    channel: str = "webchat"
    notes: str = ""


@dataclass(slots=True)
class CaseResult:
    case: EvalCase
    intent: str
    escalated: bool
    escalation_reason: str | None
    resolution_status: str
    citations: list[str]
    stubbed: bool
    latency_ms: float
    error: str | None = None

    @property
    def intent_correct(self) -> bool:
        return self.intent == self.case.expected_intent

    @property
    def escalation_correct(self) -> bool:
        return self.escalated == self.case.should_escalate

    @property
    def reason_correct(self) -> bool | None:
        """None when the case does not pin a specific reason."""
        expected = self.case.expected_escalation_reason
        if not expected or not self.escalated:
            return None
        return (self.escalation_reason or "").startswith(expected)

    @property
    def grounded(self) -> bool | None:
        """None when the case lists no expected sources."""
        if not self.case.expected_sources:
            return None
        # Citations are chunk ids ("kb-x#c1"); compare on the article prefix.
        found = {c.split("#", 1)[0] for c in self.citations}
        return bool(found & set(self.case.expected_sources))

    @property
    def deflected(self) -> bool:
        """Resolved without a human, on a case that did not need one."""
        return not self.escalated and self.resolution_status == "resolved"


def load_cases(path: Path | None = None) -> list[EvalCase]:
    source = path or DEFAULT_CASES
    with source.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    return [
        EvalCase(
            id=row["id"],
            message=row["message"],
            expected_intent=row["expected_intent"],
            should_escalate=bool(row["should_escalate"]),
            expected_sources=row.get("expected_sources", []),
            expected_escalation_reason=row.get("expected_escalation_reason"),
            customer_id=row.get("customer_id", "cust_demo"),
            customer_tier=row.get("customer_tier", "standard"),
            channel=row.get("channel", "webchat"),
            notes=row.get("notes", ""),
        )
        for row in payload["cases"]
    ]


async def run_case(graph: Any, case: EvalCase, index: int) -> CaseResult:
    """Run one case on its own thread, so cases cannot contaminate each other."""
    thread = f"eval-{index}-{case.id}"
    config = {"configurable": {"thread_id": thread}}

    state = initial_state(
        customer_id=case.customer_id,
        channel=case.channel,  # type: ignore[arg-type]
        conversation_id=thread,
        customer_tier=case.customer_tier,  # type: ignore[arg-type]
    )
    state["messages"] = [HumanMessage(content=case.message)]

    started = time.perf_counter()
    try:
        result = await graph.ainvoke(dict(state), config=config)
    except Exception as exc:
        log.warning("eval.case_failed", extra={"case": case.id, "error": str(exc)[:200]})
        return CaseResult(
            case=case,
            intent="",
            escalated=False,
            escalation_reason=None,
            resolution_status="error",
            citations=[],
            stubbed=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc)[:300],
        )

    return CaseResult(
        case=case,
        intent=str(result.get("intent", "")),
        escalated=bool(result.get("escalation_required")) or "__interrupt__" in result,
        escalation_reason=result.get("escalation_reason"),
        resolution_status=str(result.get("resolution_status", "pending")),
        citations=[c.get("id", "") for c in result.get("rag_citations", [])],
        stubbed=bool(result.get("llm_stubbed")),
        latency_ms=(time.perf_counter() - started) * 1000,
    )


async def run_suite(graph: Any, cases: list[EvalCase]) -> list[CaseResult]:
    """Run every case sequentially.

    Sequential on purpose: concurrent runs would contend on the shared free-tier
    rate buckets and distort the latency figures.
    """
    results: list[CaseResult] = []
    for index, case in enumerate(cases):
        results.append(await run_case(graph, case, index))
    return results


def score(results: list[CaseResult]) -> dict[str, Any]:
    """Aggregate case results into a report."""
    if not results:
        return {"error": "no cases run"}

    errors = [r for r in results if r.error]
    scored = [r for r in results if not r.error]

    should_resolve = [r for r in scored if not r.case.should_escalate]
    should_escalate = [r for r in scored if r.case.should_escalate]

    deflected = [r for r in should_resolve if r.deflected]

    # Escalation confusion matrix.
    true_pos = sum(1 for r in should_escalate if r.escalated)
    false_neg = sum(1 for r in should_escalate if not r.escalated)
    false_pos = sum(1 for r in should_resolve if r.escalated)
    true_neg = sum(1 for r in should_resolve if not r.escalated)

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else None
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall)
        else None
    )

    grounded = [r for r in scored if r.grounded is not None]
    reason_checked = [r for r in scored if r.reason_correct is not None]
    latencies = sorted(r.latency_ms for r in scored)
    stubbed = sum(1 for r in scored if r.stubbed)

    return {
        "cases": {
            "total": len(results),
            "scored": len(scored),
            "errored": len(errors),
        },
        "fcr": {
            "rate": _ratio(len(deflected), len(should_resolve)),
            "deflected": len(deflected),
            "eligible": len(should_resolve),
            "target": 0.92,
            "meets_target": (_ratio(len(deflected), len(should_resolve)) or 0.0) >= 0.92,
            "definition": "of cases needing no human, resolved without one",
        },
        "escalation": {
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
            "confusion": {
                "true_positive": true_pos,
                "false_negative": false_neg,
                "false_positive": false_pos,
                "true_negative": true_neg,
            },
            "reason_accuracy": _ratio(
                sum(1 for r in reason_checked if r.reason_correct), len(reason_checked)
            ),
            "note": (
                "false_negative is the serious direction: a customer who needed a "
                "human did not get one"
            ),
        },
        "intent_accuracy": _ratio(sum(1 for r in scored if r.intent_correct), len(scored)),
        "citation_grounding": _ratio(sum(1 for r in grounded if r.grounded), len(grounded)),
        "latency_ms": {
            "p50": _round(latencies[len(latencies) // 2]) if latencies else None,
            "max": _round(latencies[-1]) if latencies else None,
        },
        "data_quality": {
            "stubbed_share": _ratio(stubbed, len(scored)),
            "warning": (
                "Run against the offline stub: these figures score keyword "
                "heuristics, not model behaviour. Set GROQ_API_KEY to evaluate "
                "the real system."
            )
            if stubbed
            else None,
        },
        "failures": [
            {
                "id": r.case.id,
                "message": r.case.message[:80],
                "expected_intent": r.case.expected_intent,
                "got_intent": r.intent,
                "should_escalate": r.case.should_escalate,
                "escalated": r.escalated,
                "escalation_reason": r.escalation_reason,
                "grounded": r.grounded,
                "notes": r.case.notes,
            }
            for r in scored
            if not r.intent_correct or not r.escalation_correct or r.grounded is False
        ],
        "errors": [{"id": r.case.id, "error": r.error} for r in errors],
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
