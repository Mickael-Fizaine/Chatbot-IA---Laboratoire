import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "lab_documents")
EXPECTED_DIM    = 3072
EMBED_MODEL     = "models/gemini-embedding-001"

# Read expected count from documents.json so the test adapts to any dataset size
_DOCS_PATH = Path(__file__).parent.parent / "data" / "documents.json"
if _DOCS_PATH.exists():
    with open(_DOCS_PATH, encoding="utf-8") as _f:
        EXPECTED_POINTS = len(json.load(_f))
else:
    EXPECTED_POINTS = 20_000  # fallback : large dataset default


def check_collection(client):
    collections = {c.name for c in client.get_collections().collections}
    assert COLLECTION_NAME in collections, \
        f"Collection '{COLLECTION_NAME}' introuvable. Disponibles : {collections}"

    info   = client.get_collection(COLLECTION_NAME)
    points = info.points_count
    dim    = info.config.params.vectors.size

    print(f"\n--- Stats collection ---")
    print(f"  Nom      : {COLLECTION_NAME}")
    print(f"  Points   : {points:,}  (attendu : {EXPECTED_POINTS:,})")
    print(f"  Dim      : {dim}  (attendu : {EXPECTED_DIM})")

    assert points == EXPECTED_POINTS, \
        f"Attendu {EXPECTED_POINTS:,} points, obtenu {points:,}. " \
        f"Relancez : python ingestion/indexer.py"
    assert dim == EXPECTED_DIM, \
        f"Dimension attendue {EXPECTED_DIM}, obtenu {dim}"
    return points, dim


def check_random_samples(client):
    total = client.get_collection(COLLECTION_NAME).points_count
    ids   = random.sample(range(total), min(3, total))
    results = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=ids,
        with_payload=True,
        with_vectors=False,
    )

    print(f"\n--- 3 documents aleatoires ---")
    for point in results:
        payload = point.payload or {}
        preview = (payload.get("content", "")[:150] + "...") if payload.get("content") else "(vide)"
        print(f"  id          : {point.id}")
        print(f"  source_file : {payload.get('source_file', 'N/A')}")
        print(f"  compound    : {payload.get('compound_name', 'N/A')}")
        print(f"  content     : {preview}")
        print()


def check_vector_search(client):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    query  = "cytotoxicity test on cancer cells"
    model  = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vector = model.embed_query(query)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=3,
        with_payload=True,
    )
    hits = response.points

    print(f"--- Recherche vectorielle : '{query}' ---")
    assert len(hits) > 0, "Aucun resultat — Qdrant vide ou requete invalide"
    for i, hit in enumerate(hits, 1):
        payload = hit.payload or {}
        preview = (payload.get("content", "")[:120] + "...") if payload.get("content") else "(vide)"
        print(f"  [{i}] score={hit.score:.4f} | {payload.get('source_file', 'N/A')}")
        print(f"       {preview}")
    return len(hits)


def main():
    from qdrant_client import QdrantClient

    print("=" * 65)
    print("TEST INDEXATION — lab_documents @ Qdrant")
    print(f"Dataset attendu : {EXPECTED_POINTS:,} points  (data/documents.json)")
    print("=" * 65)

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        print(f"\nConnecte a Qdrant sur {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        print(f"\nFAIL : impossible de se connecter a Qdrant — {e}")
        print("  -> Lancez Docker : docker start qdrant")
        sys.exit(1)

    errors = []
    points = dim = n_results = 0

    try:
        points, dim = check_collection(client)
    except AssertionError as e:
        errors.append(f"Collection : {e}")

    try:
        check_random_samples(client)
    except Exception as e:
        errors.append(f"Echantillons aleatoires : {e}")

    try:
        n_results = check_vector_search(client)
    except Exception as e:
        errors.append(f"Recherche vectorielle : {e}")

    print("\n" + "=" * 65)
    if errors:
        for err in errors:
            print(f"FAIL : {err}")
        sys.exit(1)
    else:
        print(f"PASS : {points:,} points indexes (dim={dim}), recherche vectorielle OK ({n_results} resultats)")


if __name__ == "__main__":
    main()
