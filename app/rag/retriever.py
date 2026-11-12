"""Hybrid retrieval: vector search + keyword search, fused, then reranked.

1. Vector search finds passages that *mean* the same as the query.
2. Keyword search (BM25) finds passages that share its exact words — names,
   numbers and jargon that embeddings blur.
3. Reciprocal rank fusion merges the two rankings.
4. A cross-encoder reads the query against each candidate and gives the final
   order, discarding candidates it judges irrelevant.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.config import Settings
from app.db import ChunkRecord, Database
from app.logging_config import logger
from app.rag.index import VectorIndex, passage_text
from app.rag.models import Embedder, Reranker

_RRF_K = 60  # standard damping constant for reciprocal rank fusion


@dataclass(frozen=True)
class Passage:
    chunk: ChunkRecord
    score: float  # 0..1; reranker relevance when available, else fused rank score


@dataclass(frozen=True)
class RetrievalResult:
    passages: list[Passage]
    # "matched"  — passages relevant to the query
    # "overview" — nothing matched; these are the opening sections of each document
    # "empty"    — there is nothing in the library to search
    mode: str


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = _RRF_K) -> list[str]:
    """Merge ranked id lists; ids ranked highly in several lists come first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        embedder: Embedder,
        index: VectorIndex,
        reranker: Reranker | None = None,
    ):
        self._settings = settings
        self._db = db
        self._embedder = embedder
        self._index = index
        self._reranker = reranker

    def search(
        self, query: str, doc_ids: Sequence[str] | None = None, top_k: int | None = None
    ) -> RetrievalResult:
        top_k = top_k or self._settings.top_k
        pool = self._settings.retrieval_candidates

        semantic = [cid for cid, _ in self._index.search(
            self._embedder.embed_query(query), pool, doc_ids
        )]
        keyword = self._db.keyword_search(query, pool, doc_ids)
        fused = reciprocal_rank_fusion([semantic, keyword])
        # Reranking is the expensive step on a CPU; only the best fused candidates get it.
        fused = fused[: self._settings.rerank_candidates if self._reranker else top_k]
        records = self._db.get_chunks(fused)
        candidates = [records[cid] for cid in fused if cid in records]

        ranked = self._rank(query, candidates)
        floor = self._settings.rerank_min_score if self._reranker else 0.0
        matched = [p for p in ranked if p.score >= floor][:top_k]
        if matched:
            # A two-part question scores each passage against the whole question, so the
            # passage answering the second part can fall under the cut-off. Missing a
            # passage costs more than an extra one: keep the next-best few regardless.
            keep = max(len(matched), min(self._settings.min_passages, top_k))
            matched = ranked[:keep]
        logger.info(
            "Retrieval '%s': %d semantic + %d keyword -> %d candidates -> %d passages",
            query[:60], len(semantic), len(keyword), len(candidates), len(matched),
        )
        if matched:
            return RetrievalResult(matched, "matched")

        # Nothing clearly relevant (a greeting, or a broad request such as "summarise
        # my documents"). Give the LLM the opening of each document for orientation,
        # plus the best couple of candidates in case the relevance cut-off was too strict.
        overview = ranked[:2]
        seen = {p.chunk.id for p in overview}
        for chunk in self._db.leading_chunks(per_doc=2, limit=top_k, doc_ids=doc_ids):
            if chunk.id not in seen and len(overview) < top_k:
                overview.append(Passage(chunk, 0.0))
        return RetrievalResult(overview, "overview" if overview else "empty")

    def _rank(self, query: str, candidates: list[ChunkRecord]) -> list[Passage]:
        if not candidates:
            return []
        if self._reranker is None:
            # No cross-encoder: keep the fused order, scored by rank position.
            return [Passage(c, round(1.0 / (i + 1), 4)) for i, c in enumerate(candidates)]
        scores = self._reranker.score(
            query, [passage_text(c.filename, c.heading, c.text) for c in candidates]
        )
        passages = [Passage(c, round(s, 4)) for c, s in zip(candidates, scores, strict=True)]
        return sorted(passages, key=lambda p: p.score, reverse=True)
