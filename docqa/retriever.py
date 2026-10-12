"""Semantic search over the indexed chunks."""

import logging
from dataclasses import dataclass

from docqa.config import Settings

log = logging.getLogger(__name__)

# Cap on chunks sent to the LLM when a question is split into several searches.
MAX_MERGED_CHUNKS = 8


@dataclass
class Source:
    text: str
    source: str
    page: int
    score: float
    kind: str = "document"
    rerank_score: float | None = None
    page_end: int | None = None  # last page, when the chunk runs past `page`

    @property
    def pages(self) -> range:
        return range(self.page, max(self.page, self.page_end or self.page) + 1)

    @property
    def label(self) -> str:
        if len(self.pages) > 1:
            return f"{self.source}, pp. {self.page}-{self.pages[-1]}"
        return f"{self.source}, p. {self.page}"


def retrieve(
    question: str,
    *,
    store,
    namespace: str,
    settings: Settings,
    top_k: int | None = None,
    min_score: float | None = None,
    documents: list[str] | None = None,
    workspace: str = "",
) -> list[Source]:
    """Return the best-matching chunks, best first, dropping weak matches.

    `documents` restricts the search to the named files, and `workspace` to
    one user's library.
    """
    k = settings.top_k if top_k is None else top_k
    threshold = settings.min_score if min_score is None else min_score

    conditions = {}
    if workspace:
        conditions["workspace"] = workspace
    if documents:
        conditions["source"] = {"$in": documents}
    search = {"filter": conditions} if conditions else {}
    hits = store.similarity_search_with_score(question, k=k, namespace=namespace, **search)
    sources = [
        Source(
            text=doc.page_content,
            source=str(doc.metadata.get("source", "unknown")),
            page=int(doc.metadata.get("page", 1)),
            score=float(score),
            kind=str(doc.metadata.get("kind", "document")),
            page_end=int(doc.metadata.get("page_end", doc.metadata.get("page", 1))),
        )
        for doc, score in hits
        if score >= threshold
    ]
    return sorted(sources, key=lambda s: s.score, reverse=True)


def rerank(query: str, sources: list[Source], reranker, top_k: int) -> list[Source]:
    """Keep the top_k sources the reranker judges most relevant to the query.

    If the reranker fails (for example a used-up free quota), the vector-search
    order is kept, so answers degrade slightly instead of breaking.
    """
    try:
        scores = reranker(query, [s.text for s in sources])
    except Exception as error:
        log.warning("Reranker unavailable, using vector-search order: %s", error)
        return sources[:top_k]
    for source, score in zip(sources, scores):
        source.rerank_score = float(score)
    return sorted(sources, key=lambda s: s.rerank_score, reverse=True)[:top_k]


def retrieve_many(
    queries: list[str],
    *,
    settings: Settings,
    top_k: int | None = None,
    reranker=None,
    **search,
) -> list[Source]:
    """Search for each query and interleave the results, so every query is represented.

    One embedding of a two-topic question lands on a single topic; searching per
    topic and taking results in turn keeps evidence for both. With a reranker,
    each search fetches extra candidates and the reranker picks the best top_k.
    """
    k = settings.top_k if top_k is None else top_k
    fetch = max(k, settings.rerank_candidates) if reranker else k
    results = []
    for query in queries:
        hits = retrieve(query, settings=settings, top_k=fetch, **search)
        results.append(rerank(query, hits, reranker, k) if reranker and hits else hits)
    limit = k if len(queries) == 1 else min(k * len(queries), MAX_MERGED_CHUNKS)

    merged, seen = [], set()
    for rank in range(max(map(len, results), default=0)):
        for hits in results:
            if rank >= len(hits):
                continue
            key = (hits[rank].source, hits[rank].page, hits[rank].text)
            if key not in seen:
                seen.add(key)
                merged.append(hits[rank])
    return merged[:limit]
