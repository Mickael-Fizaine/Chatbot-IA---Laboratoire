from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.reranker import Reranker


def _normalize(values: list[float]) -> list[float]:
    min_v, max_v = min(values), max(values)
    span = max_v - min_v
    if span == 0:
        return [1.0] * len(values)
    return [(v - min_v) / span for v in values]


class EnsembleRetriever:
    def __init__(self):
        self._bm25 = BM25Retriever()
        self._vector = VectorRetriever()
        self._reranker = Reranker()

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        bm25_docs = self._bm25.retrieve(query, k=20)
        vector_docs = self._vector.retrieve(query, k=20, filters=filters)

        # Deduplicate by content, keeping best score per source
        seen: dict[str, dict] = {}
        for doc in bm25_docs + vector_docs:
            key = doc["content"]
            if key not in seen:
                seen[key] = dict(doc)
                seen[key]["bm25_score"] = doc["score"] if doc["source"] == "bm25" else 0.0
                seen[key]["vector_score"] = doc["score"] if doc["source"] == "vector" else 0.0
            else:
                if doc["source"] == "bm25":
                    seen[key]["bm25_score"] = max(seen[key].get("bm25_score", 0.0), doc["score"])
                else:
                    seen[key]["vector_score"] = max(seen[key].get("vector_score", 0.0), doc["score"])

        merged = list(seen.values())

        # Normalize each score distribution independently
        bm25_scores = [d["bm25_score"] for d in merged]
        vector_scores = [d["vector_score"] for d in merged]
        norm_bm25 = _normalize(bm25_scores)
        norm_vector = _normalize(vector_scores)

        for doc, nb, nv in zip(merged, norm_bm25, norm_vector):
            doc["bm25_score_norm"] = nb
            doc["vector_score_norm"] = nv
            doc["fused_score"] = 0.6 * nv + 0.4 * nb

        # Top 20 by fused score → rerank → top_k
        top20 = sorted(merged, key=lambda d: d["fused_score"], reverse=True)[:20]
        return self._reranker.rerank(query, top20, top_n=top_k)
