import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "lab_documents")


class BM25Retriever:
    def __init__(self):
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._documents = self._scroll_all(client)
        tokenized = [doc["content"].split() for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)

    def _scroll_all(self, client: QdrantClient) -> list[dict]:
        docs = []
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                payload = point.payload or {}
                content = payload.pop("content", "")
                docs.append({"content": content, "metadata": payload})
            if offset is None:
                break
        return docs

    def retrieve(self, query: str, k: int = 20) -> list[dict]:
        tokens = query.split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "content": self._documents[i]["content"],
                "metadata": self._documents[i]["metadata"],
                "score": float(scores[i]),
                "source": "bm25",
            }
            for i in top_indices
        ]
