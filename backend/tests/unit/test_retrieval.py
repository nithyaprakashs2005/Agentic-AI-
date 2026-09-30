"""Hybrid retrieval: BM25, RRF fusion, chunking, and filtering."""

from __future__ import annotations

from typing import Any

import pytest

from app.ingestion.knowledge_loader import chunk_text, load_articles, to_documents
from app.services.vector_store import (
    BM25Index,
    Document,
    HashingEmbedder,
    InMemoryHybridStore,
    reciprocal_rank_fusion,
    tokenize,
)


# --- RRF ----------------------------------------------------------------
def test_rrf_rewards_agreement_between_branches() -> None:
    """A doc ranked #1 by both branches must beat one ranked #1 by only one."""
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    assert fused["a"][0] > fused["b"][0]
    assert fused["a"][0] > fused["c"][0]


def test_rrf_score_matches_the_formula() -> None:
    fused = reciprocal_rank_fusion([["x"]], k=60)
    assert fused["x"][0] == pytest.approx(1 / 61)


def test_rrf_records_per_branch_ranks() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert fused["a"][1] == {0: 1, 1: 2}
    assert fused["b"][1] == {0: 2, 1: 1}


def test_rrf_of_nothing_is_empty() -> None:
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


# --- BM25 ---------------------------------------------------------------
def test_bm25_ranks_the_term_bearing_document_first() -> None:
    index = BM25Index()
    index.add("d1", "refund timing and card statements")
    index.add("d2", "return labels and packaging")
    scores = index.score("refund")
    assert scores["d1"] > 0
    assert "d2" not in scores


def test_bm25_ignores_stopwords() -> None:
    index = BM25Index()
    index.add("d1", "the quick brown fox")
    assert index.score("the") == {}


def test_bm25_rare_terms_outrank_common_ones() -> None:
    index = BM25Index()
    for i in range(10):
        index.add(f"common{i}", "refund policy information")
    index.add("rare", "authenticator clock drift")
    scores = index.score("authenticator refund")
    assert max(scores, key=lambda k: scores[k]) == "rare"


def test_bm25_reindex_does_not_double_count() -> None:
    index = BM25Index()
    index.add("d1", "refund refund refund")
    first = index.score("refund")["d1"]
    index.add("d1", "refund refund refund")  # same id again
    assert index.score("refund")["d1"] == pytest.approx(first)


def test_bm25_empty_query_or_corpus() -> None:
    assert BM25Index().score("anything") == {}
    index = BM25Index()
    index.add("d1", "content here")
    assert index.score("") == {}


# --- Embedder -----------------------------------------------------------
def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dim=384)
    a = embedder.encode("refund timing")
    b = embedder.encode("refund timing")
    assert a == b
    assert len(a) == 384
    assert sum(x * x for x in a) == pytest.approx(1.0, abs=1e-6)


def test_hashing_embedder_similarity_tracks_overlap() -> None:
    embedder = HashingEmbedder()
    query = embedder.encode("how long do refunds take")
    close = embedder.encode("refunds take three to five days")
    far = embedder.encode("authenticator clock drift settings")

    def cosine(x: list[float], y: list[float]) -> float:
        return sum(a * b for a, b in zip(x, y, strict=True))

    assert cosine(query, close) > cosine(query, far)


def test_tokenize_drops_stopwords_and_single_chars() -> None:
    assert tokenize("How is the a b refund?") == ["refund"]


# --- Store --------------------------------------------------------------
@pytest.fixture
async def store() -> InMemoryHybridStore:
    s = InMemoryHybridStore(HashingEmbedder())
    await s.upsert(
        [
            Document(
                "r1",
                "Refunds appear on a card within three to five business days.",
                "billing",
                title="Refund timing",
            ),
            Document(
                "r2",
                "Return items within 30 days of delivery in original packaging.",
                "returns",
                title="Return window",
            ),
            Document(
                "r3",
                "Authenticator codes fail when the device clock drifts.",
                "technical",
                title="2FA problems",
            ),
        ]
    )
    return s


async def test_search_finds_the_right_document(store: InMemoryHybridStore) -> None:
    hits = await store.search("when will my refund show up on my card", limit=3)
    assert hits[0].id == "r1"


async def test_category_filter_restricts_results(store: InMemoryHybridStore) -> None:
    hits = await store.search("refund", categories=["returns"], limit=5)
    assert all(h.payload["category"] == "returns" for h in hits)


async def test_unmatched_category_falls_back_rather_than_returning_nothing(
    store: InMemoryHybridStore,
) -> None:
    """A misrouted intent must not starve retrieval entirely."""
    hits = await store.search("refund timing", categories=["nonexistent"], limit=3)
    assert hits, "expected fallback to unfiltered search"


async def test_locale_filter(store: InMemoryHybridStore) -> None:
    hits = await store.search("refund", locale="fr-FR", limit=5)
    assert hits  # falls back rather than returning nothing


async def test_hits_carry_both_branch_ranks(store: InMemoryHybridStore) -> None:
    hits = await store.search("refund business days", limit=3)
    top = hits[0]
    assert top.dense_rank is not None or top.sparse_rank is not None


async def test_citation_shape(store: InMemoryHybridStore) -> None:
    hits = await store.search("refund", limit=1)
    citation = hits[0].citation()
    assert set(citation) == {"id", "title", "source", "category", "relevance"}


async def test_search_on_empty_store() -> None:
    empty = InMemoryHybridStore(HashingEmbedder())
    assert await empty.search("anything") == []


async def test_upsert_replaces_by_id(store: InMemoryHybridStore) -> None:
    before = store.describe()["documents"]
    await store.upsert([Document("r1", "Replaced text about refunds.", "billing")])
    assert store.describe()["documents"] == before


def test_describe_reports_backend_and_fusion(store: Any) -> None:
    described = InMemoryHybridStore(HashingEmbedder()).describe()
    assert described["backend"] == "in-memory"
    assert described["fusion"] == "rrf"


# --- Chunking -----------------------------------------------------------
def test_chunking_respects_the_word_budget() -> None:
    text = ". ".join(f"Sentence number {i} carries some filler words here" for i in range(60)) + "."
    chunks = chunk_text(text, max_words=100)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 for c in chunks), [len(c.split()) for c in chunks]


def test_chunking_keeps_short_text_whole() -> None:
    assert chunk_text("One short sentence here.") == ["One short sentence here."]


def test_chunking_never_splits_mid_sentence() -> None:
    text = " ".join(f"This is sentence {i}." for i in range(40))
    for chunk in chunk_text(text, max_words=30):
        assert chunk.strip().endswith(".")


def test_chunking_of_empty_text() -> None:
    assert chunk_text("") == []


def test_seed_corpus_loads_and_expands() -> None:
    articles = load_articles()
    assert len(articles) >= 10
    docs = to_documents(articles)
    assert len(docs) >= len(articles)
    assert all(d.category for d in docs)
    assert all(d.source_url for d in docs)


def test_seed_categories_match_the_router_vocabulary() -> None:
    """Retrieval filters `category` by `state["intent"]`, so they cannot drift."""
    from app.agents.state import INTENTS

    categories = {a["category"] for a in load_articles()}
    assert categories <= set(INTENTS), f"unknown categories: {categories - set(INTENTS)}"
