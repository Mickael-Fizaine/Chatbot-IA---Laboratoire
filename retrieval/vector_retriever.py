import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "lab_documents")
# text-embedding-004 was renamed to gemini-embedding-001
EMBED_MODEL = "models/gemini-embedding-001"


class VectorRetriever:
    def __init__(self):
        self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    def _build_filter(self, filters: dict) -> Filter | None:
        if not filters:
            return None
        conditions = [
            FieldCondition(key=field, match=MatchValue(value=value))
            for field, value in filters.items()
        ]
        return Filter(must=conditions)

    def retrieve(self, query: str, k: int = 20, filters: dict | None = None) -> list[dict]:
        query_vector = self._embeddings.embed_query(query)
        qdrant_filter = self._build_filter(filters)

        response = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results = []
        for hit in response.points:
            payload = dict(hit.payload or {})
            content = payload.pop("content", "")
            results.append({
                "content": content,
                "metadata": payload,
                "score": float(hit.score),
                "source": "vector",
            })
        return results
