"""Run the FCR evaluation suite.

    uv run python scripts/eval_fcr.py
    uv run python scripts/eval_fcr.py --json report.json
    uv run python scripts/eval_fcr.py --fail-under 0.92

Boots the real subsystems in-process -- no server required. Exits non-zero when
`--fail-under` is given and the FCR rate falls short, so this can gate CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.graph import init_graph
from app.evals.harness import load_cases, run_suite, score
from app.ingestion.knowledge_loader import ingest
from app.services.crm import init_crm
from app.services.llm import get_gateway
from app.services.session import init_sessions
from app.services.vector_store import init_store
from app.utils.logging import configure_logging
from app.utils.runtime import run as run_async


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate first-contact resolution.")
    parser.add_argument("--cases", type=Path, default=None, help="Path to a case file")
    parser.add_argument("--json", type=Path, default=None, help="Write the report as JSON")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit non-zero if the FCR rate is below this (e.g. 0.92)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print every case result")
    args = parser.parse_args()

    configure_logging("ERROR")  # the report is the output; logs are noise

    # Pace to the free-tier limit. Without this a shed call becomes stub output
    # and the report would score the stub while claiming to score the model.
    import os

    os.environ["LLM_WAIT_ON_RATE_LIMIT"] = "true"
    from app.config import get_settings as _gs

    _gs.cache_clear()

    await get_gateway().probe_ollama()
    init_crm()
    await init_sessions()
    store = await init_store()
    await ingest(store)
    graph = await init_graph()

    cases = load_cases(args.cases)
    print(f"Running {len(cases)} labelled cases...\n")

    results = await run_suite(graph, cases)
    report = score(results)

    if args.verbose:
        for result in results:
            mark = "ok  " if result.intent_correct and result.escalation_correct else "FAIL"
            print(
                f"  [{mark}] {result.case.id:26} "
                f"intent={result.intent:9} esc={result.escalated!s:5} "
                f"{result.latency_ms:6.0f}ms"
            )
        print()

    _print_report(report)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.json}")

    if args.fail_under is not None:
        rate = report["fcr"]["rate"] or 0.0
        if rate < args.fail_under:
            print(f"\nFAIL: FCR {rate:.1%} is below the {args.fail_under:.1%} gate.")
            return 1
    return 0


def _print_report(report: dict[str, Any]) -> None:
    fcr = report["fcr"]
    esc = report["escalation"]
    confusion = esc["confusion"]

    print("=" * 66)
    print("FIRST-CONTACT RESOLUTION")
    print("=" * 66)
    print(f"  rate            {_pct(fcr['rate'])}   ({fcr['deflected']}/{fcr['eligible']})")
    print(
        f"  target          {_pct(fcr['target'])}   {'MET' if fcr['meets_target'] else 'NOT MET'}"
    )
    print(f"  definition      {fcr['definition']}")

    print("\nESCALATION")
    print(f"  precision       {_pct(esc['precision'])}")
    print(f"  recall          {_pct(esc['recall'])}")
    print(f"  f1              {_pct(esc['f1'])}")
    print(f"  reason accuracy {_pct(esc['reason_accuracy'])}")
    print(
        f"  confusion       tp={confusion['true_positive']} "
        f"fn={confusion['false_negative']} "
        f"fp={confusion['false_positive']} "
        f"tn={confusion['true_negative']}"
    )
    if confusion["false_negative"]:
        print(
            f"  WARNING: {confusion['false_negative']} case(s) needed a human and did not get one"
        )

    print("\nQUALITY")
    print(f"  intent accuracy    {_pct(report['intent_accuracy'])}")
    print(f"  citation grounding {_pct(report['citation_grounding'])}")
    print(f"  latency p50/max    {report['latency_ms']['p50']} / {report['latency_ms']['max']} ms")

    quality = report["data_quality"]
    if quality["warning"]:
        print(f"\n  ** {quality['warning']}")

    if report["failures"]:
        print(f"\nFAILING CASES ({len(report['failures'])})")
        for failure in report["failures"]:
            bits = []
            if failure["expected_intent"] != failure["got_intent"]:
                bits.append(f"intent {failure['got_intent']!r} != {failure['expected_intent']!r}")
            if failure["should_escalate"] != failure["escalated"]:
                bits.append(
                    f"escalated={failure['escalated']} expected={failure['should_escalate']}"
                )
            if failure["grounded"] is False:
                bits.append("not grounded in the expected source")
            print(f"  - {failure['id']}: {'; '.join(bits)}")
            if failure["notes"]:
                print(f"      note: {failure['notes']}")

    if report["errors"]:
        print(f"\nERRORED CASES ({len(report['errors'])})")
        for err in report["errors"]:
            print(f"  - {err['id']}: {err['error']}")
    print("=" * 66)


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


if __name__ == "__main__":
    # Not asyncio.run: on Windows that yields a ProactorEventLoop, which
    # psycopg cannot use, so the PostgreSQL checkpointer would fail here.
    raise SystemExit(run_async(main()))
