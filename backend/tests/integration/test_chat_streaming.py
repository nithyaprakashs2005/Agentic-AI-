"""Token streaming on the chat SSE endpoint.

Before this, `/api/chat/stream` sent node-progress events and then the whole
reply in one `message` frame. On a real provider that is roughly twenty seconds
of spinner followed by a wall of text arriving at once -- the single worst part
of using the thing.

Two properties matter and pull against each other: the customer-facing answer
should arrive progressively, and *only* that answer should. Intent
classification, sentiment scoring and relevance judging all run through the
same gateway, and streaming those into the chat window would show the customer
the agent thinking out loud about them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytestmark = pytest.mark.timeout(120)


def _frames(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE body into (event, data) pairs."""
    out: list[tuple[str, dict[str, Any]]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            try:
                out.append((event, json.loads("\n".join(data_lines))))
            except json.JSONDecodeError:
                continue
    return out


async def _stream(api_client: Any, message: str, conversation_id: str) -> list[tuple[str, Any]]:
    response = await api_client.post(
        "/api/chat/stream",
        json={"message": message, "conversation_id": conversation_id},
    )
    assert response.status_code == 200
    return _frames(response.text)


# --- The contract ----------------------------------------------------------
async def test_the_stream_still_starts_and_finishes(api_client: Any) -> None:
    frames = await _stream(api_client, "How long do refunds take?", "stream-shape")
    events = [name for name, _ in frames]

    assert events[0] == "start"
    assert events[-1] == "done"
    assert "message" in events


async def test_the_final_message_is_still_authoritative(api_client: Any) -> None:
    """Deltas are a preview; `message` carries the channel-rendered reply."""
    frames = await _stream(api_client, "What is the return window?", "stream-final")
    message = next(data for name, data in frames if name == "message")

    assert message["reply"]
    assert "intent" in message
    assert "citations" in message


async def test_deltas_accumulate_to_the_final_reply(api_client: Any) -> None:
    """When a provider streams, the fragments must reconstruct the answer.

    Skipped on the offline stub, which returns a whole completion and so has
    nothing to stream -- that fallback is covered below.
    """
    frames = await _stream(api_client, "How long do refunds take?", "stream-accum")
    deltas = [data["text"] for name, data in frames if name == "delta"]
    if not deltas:
        pytest.skip("no streaming provider configured; stub path covered separately")

    message = next(data for name, data in frames if name == "message")
    reply = message["reply"]
    reply_text = "\n".join(reply) if isinstance(reply, list) else reply

    assert "".join(deltas).strip() == reply_text.strip()


async def test_the_stub_degrades_to_a_single_message(api_client: Any) -> None:
    """No streaming provider must not mean no answer."""
    frames = await _stream(api_client, "How long do refunds take?", "stream-stub")
    message = next(data for name, data in frames if name == "message")
    assert message["reply"], "reply is empty when nothing streamed"


async def test_only_the_answer_streams_not_the_agents_reasoning(api_client: Any) -> None:
    """Classification and relevance calls share the gateway; they must not leak.

    The router emits JSON intent objects and the retrieval verifier emits
    relevance verdicts. Either appearing in the chat window would be a defect.
    """
    frames = await _stream(api_client, "I was charged twice for my order", "stream-leak")
    streamed = "".join(data["text"] for name, data in frames if name == "delta")
    if not streamed:
        pytest.skip("no streaming provider configured")

    lowered = streamed.lower()
    for leak in ('"intent"', '"confidence"', '"relevant"', "resolution_path"):
        assert leak not in lowered, f"internal reasoning leaked into the reply: {leak}"


async def test_node_progress_still_arrives(api_client: Any) -> None:
    """Streaming tokens must not cost the "what is it doing" indicator."""
    frames = await _stream(api_client, "Where is my order ORD-84021?", "stream-nodes")
    nodes = [data["node"] for name, data in frames if name == "node"]

    assert "router" in nodes
    assert "ticketing" in nodes


async def test_a_streamed_turn_is_still_persisted(api_client: Any) -> None:
    """The graph must run identically whether or not anyone is streaming."""
    frames = await _stream(api_client, "I want to speak to a human", "stream-persist")
    message = next(data for name, data in frames if name == "message")

    assert message["ticket_id"], "a streamed turn produced no ticket"
    detail = await api_client.get(f"/api/tickets/{message['ticket_id']}")
    assert detail.status_code == 200


# --- The gateway seam ------------------------------------------------------
async def test_gateway_streams_fragments_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """`on_delta` fires per fragment, and the joined text is what is returned."""
    from app.services.llm import LLMGateway

    gateway = LLMGateway()
    fragments = ["Refunds ", "take three ", "to five days."]

    class _Chunk:
        def __init__(self, text: str) -> None:
            self.choices = [type("C", (), {"delta": type("D", (), {"content": text})()})()]
            self.usage = None

    class _Stream:
        def __aiter__(self) -> Any:
            async def gen() -> Any:
                for f in fragments:
                    yield _Chunk(f)

            return gen()

    class _LiteLLM:
        async def acompletion(self, **kwargs: Any) -> Any:
            assert kwargs["stream"] is True
            return _Stream()

    seen: list[str] = []
    text, _usage = await gateway._call_streaming(_LiteLLM(), {}, seen.append)

    assert seen == fragments
    assert text == "".join(fragments)


async def test_a_non_streaming_call_is_unchanged(api_client: Any) -> None:
    """POST /api/chat has no delta channel and must behave exactly as before."""
    response = await api_client.post(
        "/api/chat", json={"message": "How long do refunds take?", "conversation_id": "no-stream"}
    )
    assert response.status_code == 200
    assert response.json()["reply"]
