import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "lab_documents")
EXPECTED_POINTS = 300
# gemini-embedding-001 replaced text-embedding-004 and outputs 3072 dims natively
EXPECTED_DIM = 3072
EMBED_MODEL = "models/gemini-embedding-001"


def check_collection(client):
    """Step 1-3: collection exists and stats are correct."""
    collections = {c.name for c in client.get_collections().collections}
    assert COLLECTION_NAME in collections, f"Collection '{COLLECTION_NAME}' not found. Available: {collections}"

    info = client.get_collection(COLLECTION_NAME)
    points = info.points_count
    dim = info.config.params.vectors.size

    print(f"\n--- Collection stats ---")
    print(f"  Name     : {COLLECTION_NAME}")
    print(f"  Points   : {points}  (expected {EXPECTED_POINTS})")
    print(f"  Dim      : {dim}  (expected {EXPECTED_DIM})")

    assert points == EXPECTED_POINTS, f"Expected {EXPECTED_POINTS} points, got {points}"
    assert dim == EXPECTED_DIM, f"Expected dim {EXPECTED_DIM}, got {dim}"
    return points, dim


def check_random_samples(client):
    """Step 4: retrieve 3 random documents and display key fields."""
    total = client.get_collection(COLLECTION_NAME).points_count
    ids = random.sample(range(total), 3)
    results = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=ids,
        with_payload=True,
        with_vectors=False,
    )

    print(f"\n--- 3 random documents ---")
    for point in results:
        payload = point.payload or {}
        content_preview = (payload.get("content", "")[:150] + "...") if payload.get("content") else "(no content)"
        print(f"  id           : {point.id}")
        print(f"  source_file  : {payload.get('source_file', 'N/A')}")
        print(f"  domain       : {payload.get('domain', 'N/A')}")
        print(f"  content      : {content_preview}")
        print()


def check_vector_search(client):
    """Step 5: embed a query and run a top-3 similarity search."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    query = "cytotoxicity test on cancer cells"
    embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    query_vector = embeddings_model.embed_query(query)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3,
        with_payload=True,
    )
    hits = response.points

    print(f"--- Vector search: '{query}' ---")
    assert len(hits) > 0, "Vector search returned no results"
    for i, hit in enumerate(hits, 1):
        payload = hit.payload or {}
        content_preview = (payload.get("content", "")[:120] + "...") if payload.get("content") else "(no content)"
        print(f"  [{i}] score={hit.score:.4f} | {payload.get('source_file', 'N/A')}")
        print(f"       {content_preview}")
    return len(hits)


def main():
    from qdrant_client import QdrantClient

    errors = []
    points = dim = n_results = 0

    print("=" * 60)
    print("TEST INDEXATION — lab_documents @ Qdrant")
    print("=" * 60)

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        print(f"\nConnected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        print(f"\nFAIL : cannot connect to Qdrant — {e}")
        sys.exit(1)

    try:
        points, dim = check_collection(client)
    except AssertionError as e:
        errors.append(f"Collection check: {e}")

    try:
        check_random_samples(client)
    except Exception as e:
        errors.append(f"Random sample retrieval: {e}")

    try:
        n_results = check_vector_search(client)
    except Exception as e:
        errors.append(f"Vector search: {e}")

    print("\n" + "=" * 60)
    if errors:
        for err in errors:
            print(f"FAIL : {err}")
        sys.exit(1)
    else:
        print(f"PASS : collection OK, {points} points (dim={dim}), recherche vectorielle OK ({n_results} resultats)")


if __name__ == "__main__":
    main()
