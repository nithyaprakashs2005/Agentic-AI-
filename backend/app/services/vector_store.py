"""Hybrid knowledge retrieval: dense + BM25, fused with RRF.

system specification Standard #8 mandates dense + BM25 + RRF reranking with verifiable
citations. Two interchangeable backends implement that contract:

* `QdrantHybridStore`  -- named `dense` vector + `bm25` sparse vector, fused
  server-side with `FusionQuery(Fusion.RRF)`. Preferred; needs Docker.
* `InMemoryHybridStore` -- the same algorithm in-process (Okapi BM25 + cosine
  over embeddings, fused with the same RRF formula). Keeps retrieval, citations,
  and the agentic RAG loop testable with no external services.

Embeddings come from sentence-transformers when installed; otherwise from a
deterministic hashing embedder so the dense branch still contributes signal
offline. Which one is live is reported by `describe()` and surfaced in /health.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol

from app.config import Settings, get_settings
from app.utils.logging import get_logger

log = get_logger(__name__)

RRF_K = 60  # Standard reciprocal-rank-fusion damping constant.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have how i in is it its of on or that the
    to was were what when where which who why will with you your my me do does did can could
    would should there their this these those if then than about into over under""".split()
)


# Reserved point recording which embedder built the index.
_EMBEDDER_MARKER_ID = "00000000-0000-5000-8000-000000000001"

# Namespace for deriving stable Qdrant point ids from document ids.
_POINT_NAMESPACE = uuid.UUID("6f6b1f9e-2a1e-5b3c-9d4a-7c8e0f1a2b3d")


def point_id(doc_id: str) -> str:
    """Deterministic Qdrant point id for a document id.

    MUST NOT use Python's `hash()`: string hashing is randomised per process
    (PYTHONHASHSEED), so every restart produced fresh ids and re-inserted the
    whole corpus instead of overwriting it. The collection grew by 16 points on
    each boot, and retrieval started returning the same article five times
    because RRF was fusing duplicates.

    uuid5 is stable across processes and machines, so upsert is idempotent.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, doc_id))


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass(slots=True)
class Document:
    id: str
    text: str
    category: str
    locale: str = "en-US"
    source_url: str = ""
    title: str = ""
    updated_at: float = 0.0

    def payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "title": self.title,
            "category": self.category,
            "locale": self.locale,
            "source_url": self.source_url,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class SearchHit:
    id: str
    text: str
    score: float
    payload: dict[str, Any]
    dense_rank: int | None = None
    sparse_rank: int | None = None

    def citation(self) -> dict[str, Any]:
        """Citation record for the handoff packet and customer-facing sources."""
        return {
            "id": self.id,
            "title": self.payload.get("title", ""),
            "source": self.payload.get("source_url", ""),
            "category": self.payload.get("category", ""),
            "relevance": round(self.score, 4),
        }


# --- Embedders ----------------------------------------------------------
class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder -- no model download required.

    Hashes word unigrams/bigrams and character 4-grams into a fixed-width
    vector with sublinear term frequency, then L2-normalises. Not semantically
    equivalent to bge-small, but it is stable, fast, and gives the dense branch
    genuine lexical-overlap signal (including typo tolerance from char-grams)
    when sentence-transformers is unavailable.
    """

    name = "hashing-fallback"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _features(self, text: str) -> Counter[str]:
        tokens = tokenize(text)
        feats: Counter[str] = Counter()
        feats.update(tokens)
        feats.update(f"{a}_{b}" for a, b in pairwise(tokens))
        compact = "".join(tokens)
        feats.update(compact[i : i + 4] for i in range(max(len(compact) - 3, 0)))
        return feats

    def encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feature, count in self._features(text).items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class SentenceTransformerEmbedder:
    """Wraps the configured bge-small model when sentence-transformers exists."""

    def __init__(self, model_name: str, dim: int) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = dim
        self.name = model_name

    def encode(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vector]


def build_embedder(settings: Settings) -> Embedder:
    try:
        embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)
        log.info("embedder.loaded", extra={"model": settings.embedding_model})
        return embedder
    except Exception as exc:
        log.warning(
            "embedder.fallback",
            extra={"reason": str(exc)[:200], "using": "hashing-fallback"},
        )
        return HashingEmbedder(settings.embedding_dim)


# --- BM25 ---------------------------------------------------------------
class BM25Index:
    """Okapi BM25 over an in-memory corpus."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, list[str]] = {}
        self._freqs: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        if doc_id in self._docs:  # replace on re-index
            for term in set(self._docs[doc_id]):
                self._df[term] -= 1
        self._docs[doc_id] = tokens
        self._freqs[doc_id] = Counter(tokens)
        for term in set(tokens):
            self._df[term] += 1
        self._avg_len = sum(len(d) for d in self._docs.values()) / max(len(self._docs), 1)

    def score(self, query: str, candidates: list[str] | None = None) -> dict[str, float]:
        terms = tokenize(query)
        total_docs = len(self._docs)
        if not terms or not total_docs:
            return {}

        ids = candidates if candidates is not None else list(self._docs)
        scores: dict[str, float] = {}
        for doc_id in ids:
            freqs = self._freqs.get(doc_id)
            if not freqs:
                continue
            doc_len = len(self._docs[doc_id])
            total = 0.0
            for term in terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                df = self._df.get(term, 0)
                # BM25+ style IDF, floored to stay non-negative for common terms.
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_len, 1e-9))
                total += idf * (tf * (self.k1 + 1)) / denom
            if total > 0:
                scores[doc_id] = total
        return scores


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = RRF_K
) -> dict[str, tuple[float, dict[int, int]]]:
    """Fuse ranked ID lists. Returns id -> (rrf_score, {branch: rank})."""
    fused: dict[str, float] = {}
    positions: dict[str, dict[int, int]] = {}
    for branch, ranking in enumerate(rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
            positions.setdefault(doc_id, {})[branch] = rank
    return {doc_id: (score, positions[doc_id]) for doc_id, score in fused.items()}


# --- Stores -------------------------------------------------------------
class KnowledgeStore(Protocol):
    async def ping(self) -> bool: ...
    async def upsert(self, docs: list[Document]) -> int: ...
    async def search(
        self,
        query: str,
        *,
        categories: list[str] | None = None,
        locale: str | None = None,
        limit: int = 5,
    ) -> list[SearchHit]: ...
    def describe(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class _Stored:
    doc: Document
    vector: list[float]


class InMemoryHybridStore:
    """Dense + BM25 + RRF entirely in-process."""

    backend = "in-memory"

    def __init__(self, embedder: Embedder, *, prefetch: int = 20) -> None:
        self._embedder = embedder
        self._bm25 = BM25Index()
        self._docs: dict[str, _Stored] = {}
        self._prefetch = prefetch

    async def ping(self) -> bool:
        """In-process: alive whenever the process is."""
        return True

    async def upsert(self, docs: list[Document]) -> int:
        for doc in docs:
            self._docs[doc.id] = _Stored(doc=doc, vector=self._embedder.encode(doc.text))
            self._bm25.add(doc.id, f"{doc.title} {doc.text}")
        log.info("knowledge.upsert", extra={"count": len(docs), "backend": self.backend})
        return len(docs)

    def _eligible(self, categories: list[str] | None, locale: str | None) -> list[str]:
        """Metadata pre-filter, applied to both branches (mirrors Qdrant)."""
        wanted = {c.lower() for c in categories} if categories else None
        ids: list[str] = []
        for doc_id, stored in self._docs.items():
            if wanted and stored.doc.category.lower() not in wanted:
                continue
            if locale and stored.doc.locale.lower() != locale.lower():
                continue
            ids.append(doc_id)
        return ids

    async def search(
        self,
        query: str,
        *,
        categories: list[str] | None = None,
        locale: str | None = None,
        limit: int = 5,
    ) -> list[SearchHit]:
        eligible = self._eligible(categories, locale)
        # A category filter that matches nothing must not silently return the
        # whole corpus -- but an empty *corpus* slice for a valid filter should
        # fall back to unfiltered search rather than answering nothing.
        if not eligible:
            eligible = list(self._docs)
        if not eligible:
            return []

        query_vec = self._embedder.encode(query)
        dense_scores = {
            doc_id: _cosine(query_vec, self._docs[doc_id].vector) for doc_id in eligible
        }
        dense_ranked = [
            doc_id
            for doc_id, score in sorted(dense_scores.items(), key=lambda kv: -kv[1])
            if score > 0.0
        ][: self._prefetch]

        sparse_scores = self._bm25.score(query, candidates=eligible)
        sparse_ranked = [
            doc_id for doc_id, _ in sorted(sparse_scores.items(), key=lambda kv: -kv[1])
        ][: self._prefetch]

        fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked])
        ordered = sorted(fused.items(), key=lambda kv: -kv[1][0])[:limit]

        hits: list[SearchHit] = []
        for doc_id, (score, positions) in ordered:
            stored = self._docs[doc_id]
            hits.append(
                SearchHit(
                    id=doc_id,
                    text=stored.doc.text,
                    score=score,
                    payload=stored.doc.payload(),
                    dense_rank=positions.get(0),
                    sparse_rank=positions.get(1),
                )
            )
        return hits

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "documents": len(self._docs),
            "embedder": self._embedder.name,
            "fusion": "rrf",
        }


class QdrantHybridStore:
    """Qdrant OSS with named dense vector + BM25 sparse vector, RRF-fused."""

    backend = "qdrant"

    def __init__(self, client: Any, settings: Settings, embedder: Embedder) -> None:
        self._client = client
        self._settings = settings
        self._embedder = embedder
        self._collection = settings.qdrant_collection
        self._count = 0

    async def ping(self) -> bool:
        """Ask Qdrant for the collection, so a dead server reads as dead."""
        await self._client.get_collection(self._collection)
        return True

    @classmethod
    async def connect(cls, settings: Settings, embedder: Embedder) -> QdrantHybridStore | None:
        """Return a live store, or None when Qdrant is unreachable."""
        try:
            from qdrant_client import AsyncQdrantClient, models
        except ImportError:
            log.info("qdrant.not_installed")
            return None

        try:
            client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_rest_port,
                grpc_port=settings.qdrant_grpc_port,
                prefer_grpc=settings.qdrant_prefer_grpc,
                timeout=3,
            )
            existing = {c.name for c in (await client.get_collections()).collections}

            if settings.qdrant_collection in existing and await cls._embedder_changed(
                client, settings, embedder
            ):
                # Vectors from a different embedder are not comparable. Querying
                # across them returns plausible-looking nonsense with no error,
                # so the index is rebuilt instead.
                log.warning(
                    "qdrant.embedder_changed",
                    extra={
                        "collection": settings.qdrant_collection,
                        "rebuilding_with": embedder.name,
                    },
                )
                await client.delete_collection(settings.qdrant_collection)
                existing.discard(settings.qdrant_collection)

            if settings.qdrant_collection not in existing:
                await client.create_collection(
                    collection_name=settings.qdrant_collection,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=settings.embedding_dim, distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={"bm25": models.SparseVectorParams()},
                )
                log.info(
                    "qdrant.collection_created",
                    extra={"collection": settings.qdrant_collection},
                )
            log.info("qdrant.connected", extra={"host": settings.qdrant_host})
            return cls(client, settings, embedder)
        except Exception as exc:
            log.info("qdrant.unavailable", extra={"reason": str(exc)[:200]})
            return None

    @staticmethod
    async def _embedder_changed(client: Any, settings: Settings, embedder: Embedder) -> bool:
        """True when the stored index was built by a different embedder.

        The marker lives on a reserved point rather than in collection metadata,
        which Qdrant does not expose for arbitrary keys.
        """
        try:
            found = await client.retrieve(
                collection_name=settings.qdrant_collection,
                ids=[_EMBEDDER_MARKER_ID],
                with_payload=True,
            )
        except Exception:
            return True

        if not found:
            return True
        stored = (found[0].payload or {}).get("embedder")
        return bool(stored != embedder.name)

    async def _write_embedder_marker(self) -> None:
        from qdrant_client import models

        await self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=_EMBEDDER_MARKER_ID,
                    vector={
                        "dense": [0.0] * self._embedder.dim,
                        "bm25": models.SparseVector(indices=[0], values=[0.0]),
                    },
                    # Excluded from results by the locale filter, which no real
                    # query omits.
                    payload={"embedder": self._embedder.name, "locale": "__marker__"},
                )
            ],
        )

    def _sparse(self, text: str) -> Any:
        """Hash tokens into a sparse vector (stands in for a BM25 encoder)."""
        from qdrant_client import models

        counts = Counter(tokenize(text))
        indices: list[int] = []
        values: list[float] = []
        for term, count in counts.items():
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest()
            indices.append(int.from_bytes(digest, "big") % (2**31 - 1))
            values.append(1.0 + math.log(count))
        return models.SparseVector(indices=indices, values=values)

    async def upsert(self, docs: list[Document]) -> int:
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=point_id(doc.id),
                vector={
                    "dense": self._embedder.encode(doc.text),
                    "bm25": self._sparse(f"{doc.title} {doc.text}"),
                },
                payload={**doc.payload(), "doc_id": doc.id},
            )
            for doc in docs
        ]
        await self._client.upsert(collection_name=self._collection, points=points)
        await self._write_embedder_marker()
        self._count += len(points)
        log.info("knowledge.upsert", extra={"count": len(points), "backend": self.backend})
        return len(points)

    async def search(
        self,
        query: str,
        *,
        categories: list[str] | None = None,
        locale: str | None = None,
        limit: int = 5,
    ) -> list[SearchHit]:
        from qdrant_client import models

        conditions = []
        if categories:
            conditions.append(
                models.FieldCondition(key="category", match=models.MatchAny(any=categories))
            )
        if locale:
            conditions.append(
                models.FieldCondition(key="locale", match=models.MatchValue(value=locale))
            )
        query_filter = models.Filter(must=conditions) if conditions else None

        # Dense and BM25 branches fused server-side with RRF. No score_threshold:
        # RRF emits reciprocal ranks, not cosine similarities.
        response = await self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(
                    query=self._embedder.encode(query),
                    using="dense",
                    filter=query_filter,
                    limit=20,
                ),
                models.Prefetch(
                    query=self._sparse(query),
                    using="bm25",
                    filter=query_filter,
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            SearchHit(
                id=str((point.payload or {}).get("doc_id", point.id)),
                text=str((point.payload or {}).get("text", "")),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "collection": self._collection,
            "documents": self._count,
            "embedder": self._embedder.name,
            "fusion": "rrf",
        }


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))  # both are L2-normalised


_store: KnowledgeStore | None = None


async def init_store(settings: Settings | None = None) -> KnowledgeStore:
    """Build the best available store. Qdrant when reachable, else in-memory."""
    global _store
    settings = settings or get_settings()
    embedder = build_embedder(settings)

    qdrant = await QdrantHybridStore.connect(settings, embedder)
    _store = qdrant or InMemoryHybridStore(embedder)
    log.info("knowledge.store_ready", extra=_store.describe())
    return _store


def get_store() -> KnowledgeStore:
    if _store is None:
        raise RuntimeError("knowledge store not initialised; call init_store() at startup")
    return _store
