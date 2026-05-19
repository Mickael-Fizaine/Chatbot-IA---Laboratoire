"""
Reranker module — TF-IDF cosine similarity backend.

NOTE: The intended backend is FlagReranker("BAAI/bge-reranker-v2-m3") from
FlagEmbedding. Swap self._backend in __init__ once PyTorch is stable in this
environment (PyTorch 2.x segfaults on Python 3.13 / Windows CPU as of 2025-05).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Reranker:
    """Cross-encoder reranker. Currently uses TF-IDF cosine similarity."""

    def __init__(self):
        # Stateless — TfidfVectorizer is fitted per call (corpus is small)
        self.mode = "tfidf"

    def rerank(self, query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
        if not documents:
            return []

        contents = [doc["content"] for doc in documents]
        corpus = [query] + contents

        tfidf = TfidfVectorizer().fit(corpus)
        q_vec = tfidf.transform([query])
        d_vecs = tfidf.transform(contents)

        scores = cosine_similarity(q_vec, d_vecs)[0].tolist()

        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:top_n]:
            result = dict(doc)
            result["rerank_score"] = float(score)
            results.append(result)
        return results
