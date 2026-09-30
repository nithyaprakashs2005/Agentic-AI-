"""Chat endpoints: REST, SSE streaming, and WebSocket.

Three transports over one agent invocation path (`run_turn`), so a change to
orchestration behaviour cannot drift between channels. Output always passes
through `utils.channel_adapter`, never formatted here (system specification Standard #7).
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents.graph import get_graph
from app.agents.state import Channel, initial_state
from app.services.analytics import record_turn_from_state
from app.services.session import resolve_session
from app.services.tracing import span, start_trace
from app.utils.channel_adapter import render
from app.utils.logging import bind_context, get_logger, new_run_id
from app.utils.pii import mask

log = get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

_GENERIC_FAILURE = (
    "Something went wrong on our side and I could not complete that. "
    "Please try again, or ask for a human colleague and I will pass this on."
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    customer_id: str = Field(default="cust_demo", max_length=64)
    channel: Channel = "webchat"
    conversation_id: str | None = None
    customer_tier: Literal["free", "standard", "enterprise"] = "standard"


class Citation(BaseModel):
    id: str
    title: str = ""
    source: str = ""
    relevance: float = 0.0


class ChatResponse(BaseModel):
    reply: str | list[str]
    conversation_id: str
    ticket_id: str | None = None
    intent: str
    intent_confidence: float
    sentiment_score: float
    resolution_status: str
    resolution_path: str
    escalated: bool
    escalation_reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    awaiting_human: bool = False
    # Set when no LLM provider is reachable, so a client can label the reply
    # rather than presenting stub text as a model answer.
    stubbed: bool = False


class ResumeRequest(BaseModel):
    conversation_id: str
    resolution: str = Field(default="handled", max_length=2000)


async def run_turn(payload: ChatRequest) -> dict[str, Any]:
    """Invoke the orchestrator for one customer turn."""
    session = await resolve_session(
        customer_id=payload.customer_id,
        channel=payload.channel,
        conversation_id=payload.conversation_id,
    )
    run_id = new_run_id()
    bind_context(
        conversation_id=session.conversation_id,
        customer_id=payload.customer_id,
        channel=payload.channel,
        langgraph_run_id=run_id,
    )

    graph = get_graph()
    config = {"configurable": {"thread_id": session.conversation_id}}

    # First turn on a thread seeds the full state; later turns append a message
    # and let the checkpointer supply everything else.
    snapshot = await graph.aget_state(config)
    if snapshot.values:
        payload_state: dict[str, Any] = {
            "messages": [HumanMessage(content=payload.message)],
            # Refreshed every turn: a customer who moves from chat to phone must
            # widen the ticket's channel list, not start a second ticket.
            "channel": payload.channel,
            "channels": list(session.channels),
        }
    else:
        seeded = initial_state(
            customer_id=payload.customer_id,
            channel=payload.channel,
            conversation_id=session.conversation_id,
            customer_tier=payload.customer_tier,
        )
        seeded["messages"] = [HumanMessage(content=payload.message)]
        seeded["channels"] = list(session.channels)
        payload_state = dict(seeded)

    masked = mask(payload.message)
    log.info(
        "chat.turn_received",
        extra={
            "channel": payload.channel,
            "pii_kinds": masked.kinds,
            "chars": len(payload.message),
        },
    )

    started = time.perf_counter()
    with (
        start_trace() as trace_id,
        span("chat.turn", kind="request", channel=payload.channel, run_id=run_id),
    ):
        result = await graph.ainvoke(payload_state, config=config)
    total_ms = (time.perf_counter() - started) * 1000

    record_turn_from_state(conversation_id=session.conversation_id, state=result, total_ms=total_ms)

    awaiting_human = "__interrupt__" in result
    reply_text = result.get("draft_response") or ""
    if not reply_text:
        reply_text = _GENERIC_FAILURE

    session.ticket_id = result.get("ticket_id")
    return {
        "trace_id": trace_id,
        "state": result,
        "reply_text": reply_text,
        "conversation_id": session.conversation_id,
        "awaiting_human": awaiting_human,
    }


def _to_response(turn: dict[str, Any], channel: Channel) -> ChatResponse:
    state = turn["state"]
    return ChatResponse(
        reply=render(turn["reply_text"], channel),
        conversation_id=turn["conversation_id"],
        ticket_id=state.get("ticket_id"),
        intent=state.get("intent", "general"),
        intent_confidence=float(state.get("intent_confidence", 0.0)),
        sentiment_score=float(state.get("sentiment_score", 0.0)),
        resolution_status=state.get("resolution_status", "pending"),
        resolution_path=state.get("resolution_path", "simple"),
        escalated=bool(state.get("escalation_required")) or turn["awaiting_human"],
        escalation_reason=state.get("escalation_reason"),
        citations=[Citation(**c) for c in state.get("rag_citations", [])],
        awaiting_human=turn["awaiting_human"],
        stubbed=bool(state.get("llm_stubbed")),
    )


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    """Single-shot turn."""
    try:
        turn = await run_turn(payload)
    except Exception:
        log.exception("chat.turn_failed")
        # Never leak a stack trace or graph state to a customer.
        raise HTTPException(status_code=500, detail=_GENERIC_FAILURE) from None
    return _to_response(turn, payload.channel)


@router.post("/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """SSE stream: node progress, answer tokens, then the final message.

    Three event kinds, in the order a client sees them:

    * `node`   -- a graph step finished, so the UI can say what is happening.
    * `delta`  -- a fragment of the customer-facing answer, as the model emits
                  it. Only the answer is streamed; the classification and
                  relevance calls run through the same gateway and piping those
                  through would leak the agent's internal reasoning.
    * `message` -- the authoritative reply plus metadata. It repeats the full
                  text because the channel adapter reformats it (voice
                  plaintext, SMS segmentation), so the accumulated deltas are a
                  preview and this is the final form. Clients replace rather
                  than append.

    A provider with no streaming support, or the offline stub, simply emits no
    deltas and the client renders on `message` as before.
    """

    async def event_stream() -> Any:
        try:
            session = await resolve_session(
                customer_id=payload.customer_id,
                channel=payload.channel,
                conversation_id=payload.conversation_id,
            )
            graph = get_graph()
            config = {"configurable": {"thread_id": session.conversation_id}}

            snapshot = await graph.aget_state(config)
            if snapshot.values:
                state_in: dict[str, Any] = {"messages": [HumanMessage(content=payload.message)]}
            else:
                seeded = initial_state(
                    customer_id=payload.customer_id,
                    channel=payload.channel,
                    conversation_id=session.conversation_id,
                    customer_tier=payload.customer_tier,
                )
                seeded["messages"] = [HumanMessage(content=payload.message)]
                state_in = dict(seeded)

            yield _sse("start", {"conversation_id": session.conversation_id})

            final: dict[str, Any] = {}
            async for mode, chunk in graph.astream(
                state_in, config=config, stream_mode=["updates", "custom"]
            ):
                if mode == "custom":
                    # Token fragments written by the resolution node.
                    if isinstance(chunk, dict) and chunk.get("type") == "answer_delta":
                        yield _sse("delta", {"text": chunk.get("text", "")})
                    continue

                for node, update in chunk.items():
                    if node == "__interrupt__":
                        yield _sse("awaiting_human", {"node": node})
                        continue
                    final.update(update or {})
                    yield _sse("node", {"node": node, "status": "done"})

            reply = final.get("draft_response") or _GENERIC_FAILURE
            yield _sse(
                "message",
                {
                    "reply": render(reply, payload.channel),
                    "ticket_id": final.get("ticket_id"),
                    "intent": final.get("intent"),
                    "escalated": bool(final.get("escalation_required")),
                    "citations": final.get("rag_citations", []),
                    "stubbed": bool(final.get("llm_stubbed")),
                },
            )
            yield _sse("done", {})
        except Exception:
            log.exception("chat.stream_failed")
            yield _sse("error", {"detail": _GENERIC_FAILURE})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/resume", response_model=ChatResponse)
async def resume(payload: ResumeRequest) -> ChatResponse:
    """Resume a conversation parked on the HITL interrupt (human agent action)."""
    graph = get_graph()
    config = {"configurable": {"thread_id": payload.conversation_id}}

    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown conversation")
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="Conversation is not awaiting a human")

    try:
        result = await graph.ainvoke(Command(resume=payload.resolution), config=config)
    except Exception:
        log.exception("chat.resume_failed")
        raise HTTPException(status_code=500, detail=_GENERIC_FAILURE) from None

    channel: Channel = result.get("channel", "webchat")
    turn = {
        "state": result,
        "reply_text": result.get("draft_response") or "Handled by a human colleague.",
        "conversation_id": payload.conversation_id,
        "awaiting_human": "__interrupt__" in result,
    }
    return _to_response(turn, channel)


@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket) -> None:
    """Bidirectional chat. Each inbound frame is one customer turn."""
    await websocket.accept()
    conversation_id: str | None = None
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"message": raw}

            if conversation_id and not data.get("conversation_id"):
                data["conversation_id"] = conversation_id

            try:
                payload = ChatRequest(**data)
            except Exception as exc:
                await websocket.send_json({"type": "error", "detail": f"Invalid payload: {exc}"})
                continue

            try:
                turn = await run_turn(payload)
            except Exception:
                log.exception("chat.ws_turn_failed")
                await websocket.send_json({"type": "error", "detail": _GENERIC_FAILURE})
                continue

            conversation_id = turn["conversation_id"]
            response = _to_response(turn, payload.channel)
            await websocket.send_json({"type": "message", **response.model_dump()})
    except WebSocketDisconnect:
        log.info("chat.ws_disconnected", extra={"conversation_id": conversation_id})


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
