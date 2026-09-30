"""Agentic RAG node: hybrid retrieval, self-verification, multi-hop retry.

system specification Graph Nodes #3 / Standard #8. The loop is:

  1. Hybrid search (dense + BM25 + RRF), filtered to the routed intent.
  2. An SLM judges the retrieved chunks for genuine usefulness, in one call.
  3. If fewer than two survive, rewrite the query and retry once, unfiltered --
     a wrong intent classification should not starve retrieval.

Every surviving chunk contributes a citation, so no answer is ever ungrounded.
"""

from __future__ import annotations

import time
from typing import Any

from app.agents.prompts.system import QUERY_REWRITE_PROMPT, RELEVANCE_PROMPT
from app.agents.state import SupportAgentState, is_smalltalk, last_user_text
from app.services.llm import get_gateway
from app.services.vector_store import SearchHit, get_store
from app.utils.logging import get_logger
from app.utils.pii import mask

log = get_logger(__name__)

MIN_VERIFIED_CHUNKS = 2
FIRST_HOP_LIMIT = 5
SECOND_HOP_LIMIT = 3


async def agentic_rag_node(state: SupportAgentState) -> dict[str, Any]:
    started = time.perf_counter()
    question = last_user_text(state)
    # A greeting has nothing to retrieve; searching it only yields noise
    # citations and burns SLM calls.
    if not question.strip() or is_smalltalk(question):
        return {"rag_context": "", "rag_citations": [], "rag_hops": 0}

    masked_question = mask(question).text
    store = get_store()
    intent = state.get("intent", "general")

    # --- Hop 1: filtered to the routed intent ---
    hits = await store.search(
        masked_question, categories=[intent], locale="en-US", limit=FIRST_HOP_LIMIT
    )
    verified = await verify_relevance(masked_question, hits)
    hops = 1

    # --- Hop 2: rewrite and widen ---
    if len(verified) < MIN_VERIFIED_CHUNKS:
        rewritten = await _rewrite_query(masked_question, verified)
        log.info(
            "rag.second_hop",
            extra={"verified_after_hop1": len(verified), "rewritten": rewritten[:120]},
        )
        # Unfiltered: hop 1 already tested the intent hypothesis.
        extra_hits = await store.search(rewritten, categories=None, limit=SECOND_HOP_LIMIT)
        seen = {hit.id for hit in verified}
        fresh = [hit for hit in extra_hits if hit.id not in seen]
        verified.extend(await verify_relevance(rewritten, fresh))
        hops = 2

    verified.sort(key=lambda hit: -hit.score)
    context = "\n\n".join(f"[{hit.payload.get('title', hit.id)}]\n{hit.text}" for hit in verified)
    citations = [hit.citation() for hit in verified]

    elapsed = round((time.perf_counter() - started) * 1000, 2)
    log.info(
        "rag.retrieved",
        extra={
            "intent": intent,
            "hops": hops,
            "chunks": len(verified),
            "latency_ms": elapsed,
        },
    )

    return {
        "rag_context": context,
        "rag_citations": citations,
        "rag_hops": hops,
        "node_timings_ms": {"agentic_rag": elapsed},
    }


async def verify_relevance(question: str, hits: list[SearchHit]) -> list[SearchHit]:
    """Judge all candidates in one call, keeping the ones an SLM finds useful.

    Originally one call per hit -- up to five per hop, ten per turn. That is
    wasteful everywhere and fatal on a free tier: Groq allows 30 requests a
    minute and Gemini 20, so a handful of concurrent conversations exhausted the
    quota and every judgement silently degraded to the fallback.

    Judging together is also better judging: the model compares candidates
    against each other rather than rating each in isolation.
    """
    if not hits:
        return []

    numbered = "\n\n".join(
        f"[{index}] {hit.payload.get('title', hit.id)}\n{hit.text[:900]}"
        for index, hit in enumerate(hits, start=1)
    )

    result = await get_gateway().complete(
        [
            {"role": "system", "content": RELEVANCE_PROMPT},
            {
                "role": "user",
                "content": f"Customer question: {question}\n\nExcerpts:\n{numbered}",
            },
        ],
        tier="slm",
        purpose="verify_relevance",
        temperature=0.0,
        max_tokens=400,
        json_mode=True,
    )

    parsed = result.json(default=None)
    if not isinstance(parsed, dict) or "relevant" not in parsed:
        # A judge that did not answer must not silently discard retrieval: keep
        # everything and let the grounded answer prompt sort it out.
        log.warning("rag.verify_unparsed", extra={"candidates": len(hits)})
        return list(hits)

    keep = parse_relevant_indices(parsed.get("relevant"), len(hits))
    return [hit for index, hit in enumerate(hits, start=1) if index in keep]


def parse_relevant_indices(value: Any, count: int) -> set[int]:
    """Read the returned index list, tolerating strings and stray values."""
    if isinstance(value, bool):
        # Some models answer the older boolean shape; treat true as "all".
        return set(range(1, count + 1)) if value else set()

    indices: set[int] = set()
    if isinstance(value, list):
        for item in value:
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= count:
                indices.add(index)
    return indices


async def _rewrite_query(question: str, kept: list[SearchHit]) -> str:
    already = ", ".join(hit.payload.get("title", "") for hit in kept) or "nothing useful"
    result = await get_gateway().complete(
        [
            {"role": "system", "content": QUERY_REWRITE_PROMPT},
            {
                "role": "user",
                "content": f"Customer question: {question}\n\nAlready retrieved: {already}",
            },
        ],
        tier="slm",
        purpose="rewrite_query",
        temperature=0.3,
        max_tokens=200,
    )
    rewritten = result.text.strip().strip('"').split("\n")[0]
    return rewritten or question
