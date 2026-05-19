import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "lab_documents")
EMBEDDING_DIM = 3072  # gemini-embedding-001 native output dimension
BATCH_SIZE = 10
MAX_RETRIES = 5
INTER_BATCH_SLEEP = 1.0   # seconds between batches to stay under RPM quota
RATE_LIMIT_SLEEP = 60     # seconds to wait on 429


def embed_with_retry(embeddings_model: GoogleGenerativeAIEmbeddings, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with backoff — longer wait on 429 rate-limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return embeddings_model.embed_documents(texts)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            wait = RATE_LIMIT_SLEEP * (attempt + 1) if is_rate_limit else 2 ** attempt
            tqdm.write(f"  [retry {attempt + 1}/{MAX_RETRIES}] {'rate-limit' if is_rate_limit else 'error'} — waiting {wait}s")
            time.sleep(wait)


def recreate_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Dropped existing collection '{COLLECTION_NAME}'.")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"  Created collection '{COLLECTION_NAME}' (dim={EMBEDDING_DIM}, Cosine).")


def main():
    docs_path = Path(__file__).parent.parent / "data" / "documents.json"
    with open(docs_path, encoding="utf-8") as f:
        documents = json.load(f)
    print(f"Loaded {len(documents)} documents from {docs_path.name}")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    recreate_collection(client)

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    print(f"Embedding model ready: gemini-embedding-001")
    print(f"Batch size: {BATCH_SIZE} | Inter-batch sleep: {INTER_BATCH_SLEEP}s\n")

    total = len(documents)
    indexed = 0

    with tqdm(total=total, unit="doc", desc="Indexing") as pbar:
        for batch_start in range(0, total, BATCH_SIZE):
            batch = documents[batch_start : batch_start + BATCH_SIZE]
            texts = [doc["content"] for doc in batch]

            vectors = embed_with_retry(embeddings_model, texts)

            points = [
                PointStruct(
                    id=batch_start + i,
                    vector=vector,
                    payload={**doc["metadata"], "content": doc["content"]},
                )
                for i, (doc, vector) in enumerate(zip(batch, vectors))
            ]

            client.upsert(collection_name=COLLECTION_NAME, points=points)
            indexed += len(points)
            pbar.update(len(points))

            time.sleep(INTER_BATCH_SLEEP)

    print(f"\nIndexation terminee : {indexed} documents dans Qdrant")


if __name__ == "__main__":
    main()
