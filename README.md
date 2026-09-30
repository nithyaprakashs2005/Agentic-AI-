# OmniSupport AI Agent — Multi-Channel Voice & Chat Support Platform

A multi-channel AI customer support platform. A LangGraph orchestrator routes
each inbound turn through intent classification, identity verification, hybrid
retrieval with self-verification, resolution, sentiment scoring, and CRM
ticketing — escalating to a human with a full context packet when its own rules
say it should.

Built on high-performance open-source components and state-of-the-art agentic workflows.

---

## Quick start

No Docker, no API keys, no database required.

```bash
cd backend
uv sync
uv run python run.py            # add --reload for development
```

> Use `run.py`, not the bare `uvicorn` CLI. On Windows uvicorn forces an event
> loop psycopg cannot use, which silently downgrades PostgreSQL checkpointing to
> in-memory.

Open <http://localhost:8000/docs>, or bring up the web console:

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

Console routes available:
- **`/chat`**: Interactive web chat with live SSE token streaming, citations, and agent execution tracer.
- **`/voice`**: Push-to-talk voice console using browser Web Speech API (STT & TTS) with live voice-adapter text formatting.
- **`/channels`**: Channel connectivity hub with live health checks and webhook configuration for SMS, WhatsApp, Voice, and Email.
- **`/dashboard`**: Operator handoff queue with SLA countdowns, full CRM context packets, and operational metrics.
- **`/tickets`**: Searchable ticket management lifecycle with CSAT recording.

Then verify the whole suite:

```bash
cd backend && uv run python scripts/smoke.py
```

That exercises 40 behaviours end to end — grounded answers, intent routing, the
escalation rule engine, the handoff queue, HITL resume, SMS segmentation, voice
rendering, PII masking, and SSE streaming.

---

## What actually runs right now

The server boots with every backing service absent and degrades each subsystem
rather than failing. `GET /health` and `GET /api/voice/status` name every active fallback in prose.

| Subsystem | Preferred | Fallback when absent |
|:---|:---|:---|
| LLM | Groq → Gemini (`gemini-2.5-flash`) → Ollama | deterministic offline stub, flagged `stubbed: true` |
| Retrieval | Qdrant hybrid (dense + BM25 + RRF) | in-process hybrid, same RRF algorithm |
| Embeddings | `bge-small-en-v1.5` (**installed**) | feature-hashing embedder |
| Sessions | Valkey / Redis | in-process TTL dict |
| Checkpoints | `AsyncPostgresSaver` | `InMemorySaver` |
| CRM | provider API via MCP | seeded dataset (4 customers, 64 mock orders, 4 invoices) |
| Voice | LiveKit + Pipecat (Kokoro TTS / Whisper) | Browser Web Speech API (`/voice`) with voice adapter |

Two fallbacks change **behaviour**, not just durability: the LLM stub (keyword
heuristics, not a model) and the hashing embedder (lexical overlap, not
semantics). Answer quality observed with no keys set is a floor, not a forecast.

To run with real services:

```bash
cp .env.example .env      # add GROQ_API_KEY for a free-tier model
make services             # Qdrant, Valkey, Postgres, LiveKit
```

---

## Architecture

```
                    ┌──────────────────────────────────┐
  webchat / SMS ──► │  FastAPI: REST · SSE · WebSocket │
  WhatsApp / email  └────────────────┬─────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │  LangGraph StateGraph (SupportAgentState)    │
              │                                             │
              │  router ─► auth ─┬─(challenge)──────► END    │
              │                  ├─(escalate)─► escalation   │
              │                  └─(ok)─► agentic_rag        │
              │                              │               │
              │            ┌─(simple)────────┴─(complex)─┐   │
              │            ▼                             ▼   │
              │   simple_resolution           complex_resolution
              │            └──────────┬──────────────────┘   │
              │                   sentiment                  │
              │              ┌────────┴────────┐             │
              │        escalation           ticketing        │
              │              └──► ticketing ──┬──► END       │
              │                               └──► human_handoff
              │                                    (interrupt)│
              └─────────────────────────────────────────────┬─┘
                                                            │
   LLMGateway (Groq→Gemini→Ollama→stub) · MCP tools · Qdrant hybrid RAG
```

The orchestrator features an auth gate that validates sessions, a challenge branch for step-up auth,
and a single HITL pause placed after ticketing so a human operator finds a complete CRM record
waiting in the queue (implemented in `backend/app/agents/graph.py`).

### Key invariants

- **Every answer is grounded and cited.** Retrieval runs hybrid dense + BM25
  search fused with RRF, an SLM judges each hit, and unsatisfied queries get one
  rewritten second hop. Citations ride on the response and the ticket.
- **Escalation is deterministic.** Sentiment scoring is a model call; the
  decision is pure code in `evaluate_escalation()`, so all five rules are
  boundary-tested and never vary with temperature.
- **PII never reaches a model, a log, or storage.** `utils/pii.py` masks to
  reversible synthetic tokens before inference; transcripts persist scrubbed.
  Order numbers and non-Luhn references are deliberately left intact so lookups
  keep working.
- **Channel formatting lives in exactly one place.** `utils/channel_adapter.py`
  renders voice plaintext, SMS segments, WhatsApp markup, and web markdown. The
  graph never formats.
- **Tool faults degrade one turn, not the graph.** MCP calls are capped at 3s and
  return errors as data.
- **Metrics never flatter the system.** FCR counts whole conversations, not
  turns; CSAT comes only from submitted ratings, never inferred from sentiment;
  and every response reports what share of its sample ran on the offline stub.

---

## Commands

```bash
make help            # every target
make api             # backend with reload
make web             # frontend
make test            # 556 tests
make cov             # coverage gate (80% required, 88% actual)
make check           # lint + types + tests
make services        # Qdrant, Valkey, Postgres, LiveKit
make mcp             # list MCP servers and their tools
make voice-check     # voice readiness and latency budget
make smoke           # end-to-end against a running API
make eval            # measure FCR against the labelled case set
make migrate         # apply database migrations to DATABASE_URL
```

---

## Layout

```
backend/
  app/
    agents/        graph.py, state.py, nodes/, prompts/
    api/           routes/ (chat, tickets, voice, webhooks, email, analytics, health)
    mcp/           tools.py (canonical bodies) + 4 FastMCP servers
    services/      llm.py, vector_store.py, crm.py, session.py, twilio/email services
    voice/         pipeline.py (config only), latency.py
    ingestion/     chunking + upsert
    utils/         pii.py, channel_adapter.py, logging.py
  tests/           unit/ + integration/ (556 tests)
  scripts/smoke.py
frontend/
  src/
    app/           chat/, voice/, channels/, dashboard/, tickets/
    components/    chat/, voice/, channels/, dashboard/, tickets/, ui/ (Design Tokens & UI Kit)
    lib/           agent.ts, api.ts, speech.ts, time.ts
voice-worker/      Pipecat pipeline worker
```

---

## Status

**Working and verified:** the orchestrator graph and all its nodes, hybrid
retrieval with citations, the escalation rule engine, handoff packet generation,
the CRM ticket lifecycle, HITL interrupt and resume, cross-channel session
continuity, PII masking, channel adaptation, the MCP tool layer with timeouts,
REST + SSE + WebSocket transports, operational analytics, browser voice console,
channels connection board, and the web console.
556 tests, 88% coverage, `ruff` and `mypy --strict` clean.

**Security & Validation:** Inbound Twilio SMS/WhatsApp webhooks and inbound email
feature cryptographic signature validation before any payload reaches agent code,
ensuring forged or replayed requests are rejected.

**Persistence:** tickets, CSAT responses, LangGraph checkpoints, sessions, and
the knowledge base all persist across restarts (PostgreSQL, Valkey, Qdrant).
