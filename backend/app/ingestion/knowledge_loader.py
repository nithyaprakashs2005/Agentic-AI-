"""Knowledge base ingestion: load, chunk, embed, upsert.

system specification Phase 3 calls for semantic chunking at 200-400 tokens. Articles are
split on sentence boundaries so a chunk never ends mid-sentence, with a one
sentence overlap to preserve context across the seam.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from app.services.vector_store import Document, KnowledgeStore
from app.utils.logging import get_logger

log = get_logger(__name__)

# Rough token proxy: whitespace words * 1.3 approximates BPE tokens for prose.
_WORDS_PER_CHUNK_MAX = 300
_WORDS_PER_CHUNK_MIN = 60
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

DEFAULT_KB_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge_base.json"


def chunk_text(text: str, *, max_words: int = _WORDS_PER_CHUNK_MAX) -> list[str]:
    """Split prose into sentence-aligned chunks with one sentence of overlap."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    count = 0

    for sentence in sentences:
        words = len(sentence.split())
        if current and count + words > max_words:
            chunks.append(" ".join(current))
            # Carry the last sentence forward so context spans the boundary.
            current = [current[-1]] if len(current) > 1 else []
            count = len(current[0].split()) if current else 0
        current.append(sentence)
        count += words

    if current:
        tail = " ".join(current)
        tail_words = len(tail.split())
        # Fold a runt tail back into the previous chunk -- but only when the
        # result still fits the budget, or folding would produce an oversized
        # chunk that defeats the point of chunking.
        fits = chunks and (len(chunks[-1].split()) + tail_words) <= max_words
        if tail_words < _WORDS_PER_CHUNK_MIN and fits:
            chunks[-1] = f"{chunks[-1]} {tail}"
        else:
            chunks.append(tail)
    return chunks


def load_articles(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or DEFAULT_KB_PATH
    if not source.exists():
        log.warning("knowledge.source_missing", extra={"path": str(source)})
        return []
    with source.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{source} must contain a JSON array of articles")
    return data


def to_documents(articles: list[dict[str, Any]]) -> list[Document]:
    """Expand articles into chunk-level Documents carrying article provenance."""
    now = time.time()
    docs: list[Document] = []
    for article in articles:
        chunks = chunk_text(article["text"])
        for index, chunk in enumerate(chunks):
            suffix = f"#c{index}" if len(chunks) > 1 else ""
            docs.append(
                Document(
                    id=f"{article['id']}{suffix}",
                    text=chunk,
                    title=article.get("title", ""),
                    category=article.get("category", "general"),
                    locale=article.get("locale", "en-US"),
                    source_url=article.get("source_url", ""),
                    updated_at=float(article.get("updated_at", now)),
                )
            )
    return docs


async def ingest(store: KnowledgeStore, path: Path | None = None) -> dict[str, int]:
    """Load the seed corpus into `store`. Idempotent -- upserts by chunk id."""
    articles = load_articles(path)
    docs = to_documents(articles)
    if not docs:
        log.warning("knowledge.nothing_to_ingest")
        return {"articles": 0, "chunks": 0}
    count = await store.upsert(docs)
    log.info("knowledge.ingested", extra={"articles": len(articles), "chunks": count})
    return {"articles": len(articles), "chunks": count}
