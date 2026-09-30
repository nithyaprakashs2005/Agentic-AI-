"""End-to-end smoke test against a running API.

Usage:
    uv run uvicorn app.main:app --port 8000     # in one shell
    uv run python scripts/smoke.py              # in another

Exercises the paths that matter: a grounded FAQ answer, intent routing, the
escalation rule engine, the handoff queue, the HITL resume, SMS segmentation,
voice rendering, and PII masking. Exits non-zero on the first failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

PASS = "PASS"
FAIL = "FAIL"

_failures: list[str] = []


def call(base: str, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{base}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        _failures.append(label)


def chat(base: str, message: str, thread: str, **extra: Any) -> dict[str, Any]:
    status, body = call(
        base,
        "POST",
        "/api/chat",
        {"message": message, "customer_id": "cust_demo", "conversation_id": thread, **extra},
    )
    if status != 200:
        print(f"  [{FAIL}] chat({message[:30]!r}) returned {status}: {body}")
        _failures.append(f"chat {message[:30]}")
        return {}
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print(f"\nSmoke test against {base}\n" + "=" * 62)

    # --- Health ---
    print("\nHealth")
    status, health = call(base, "GET", "/health")
    check("GET /health responds 200", status == 200)
    if status != 200:
        print("\nAPI unreachable. Start it with: uv run uvicorn app.main:app --port 8000")
        return 1
    check("reports a status", bool(health.get("status")))
    check("enumerates subsystems", len(health.get("subsystems", {})) >= 4)
    print(
        f"       mode: {health['status']}  ({len(health.get('degradations', []))} degradation(s))"
    )
    for line in health.get("degradations", []):
        print(f"         - {line}")

    # --- Grounded answers and routing ---
    print("\nGrounded answers and intent routing")
    cases = [
        ("How long do refunds take to appear?", "billing", "sm-1"),
        ("What is the return window for an unopened item?", "returns", "sm-2"),
        ("My authenticator codes are being rejected", "technical", "sm-3"),
        ("Where is my order ORD-84021?", "orders", "sm-4"),
    ]
    for message, expected_intent, thread in cases:
        body = chat(base, message, thread)
        if not body:
            continue
        check(
            f"{expected_intent:9} intent for {message[:34]!r}",
            body["intent"] == expected_intent,
            f"got {body.get('intent')}",
        )
        check(f"{expected_intent:9} answer carries citations", bool(body["citations"]))
        check(f"{expected_intent:9} ticket created", bool(body["ticket_id"]))

    # --- Greeting must not escalate ---
    print("\nSmalltalk handling")
    body = chat(base, "hello there", "sm-greet")
    check("greeting does not fetch a human", body.get("escalated") is False)
    check("greeting retrieves nothing", body.get("citations") == [])

    # --- Escalation ---
    print("\nEscalation and human handoff")
    body = chat(base, "I want to speak to a human", "sm-esc")
    check("explicit request escalates", body.get("escalated") is True)
    check(
        "reason is the keyword rule",
        body.get("escalation_reason") == "explicit_keyword:human",
        str(body.get("escalation_reason")),
    )
    check("run parks awaiting a human", body.get("awaiting_human") is True)

    status, queue = call(base, "GET", "/api/tickets/queue")
    check("handoff queue is populated", status == 200 and queue.get("count", 0) >= 1)
    if queue.get("queue"):
        packet = queue["queue"][0].get("handoff_packet") or {}
        for key in (
            "executive_summary",
            "actions_attempted",
            "recommended_actions",
            "sentiment_trajectory",
            "churn_risk",
            "transcript",
        ):
            check(f"packet has {key}", key in packet)

    status, resumed = call(
        base,
        "POST",
        "/api/chat/resume",
        {"conversation_id": "sm-esc", "resolution": "handled by agent Dana"},
    )
    check("HITL resume succeeds", status == 200)
    check("resume clears the pause", resumed.get("awaiting_human") is False)

    # --- Channel adaptation ---
    print("\nChannel adaptation")
    body = chat(base, "What is the return window?", "sm-sms", channel="sms")
    segments = body.get("reply")
    check("SMS reply is a segment list", isinstance(segments, list))
    if isinstance(segments, list):
        check("every segment fits 160 chars", all(len(s) <= 160 for s in segments))

    body = chat(base, "How long do refunds take?", "sm-voice", channel="voice")
    spoken = body.get("reply", "")
    check("voice reply is a string", isinstance(spoken, str))
    check(
        "voice reply has no markdown or URLs",
        not any(token in str(spoken) for token in ("**", "##", "http", "$", "@")),
    )

    status, preview = call(
        base,
        "POST",
        "/api/voice/preview",
        {"text": "Your refund of **$247.50**. See https://example.com/x"},
    )
    check(
        "voice preview strips everything unspeakable",
        status == 200 and not any(preview.get("checks", {}).values()),
    )
    check("currency is spoken as words", "247 dollars and 50 cents" in preview.get("spoken", ""))

    # --- PII ---
    print("\nPII masking")
    body = chat(base, "my card 4111 1111 1111 1111 was charged twice", "sm-pii")
    check("card number absent from the reply", "4111 1111 1111 1111" not in json.dumps(body))
    if body.get("ticket_id"):
        _, ticket = call(base, "GET", f"/api/tickets/{body['ticket_id']}")
        check(
            "card number absent from the stored transcript",
            "4111 1111 1111 1111" not in json.dumps(ticket),
        )
        check("masked token stored instead", "CC_1" in json.dumps(ticket))

    # --- Streaming ---
    print("\nSSE streaming")
    request = urllib.request.Request(
        f"{base}/api/chat/stream",
        data=json.dumps(
            {
                "message": "How long do refunds take?",
                "customer_id": "cust_demo",
                "conversation_id": "sm-sse",
            }
        ).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        stream = response.read().decode()
    for event in ("event: start", "event: node", "event: message", "event: done"):
        check(f"stream emits {event!r}", event in stream)

    # --- Summary ---
    print("\n" + "=" * 62)
    if _failures:
        print(f"FAILED: {len(_failures)} check(s)")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
