import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.ensemble_retriever import EnsembleRetriever

QUERIES = [
    {
        "label": "Query 1 (semantique)",
        "text": "What are the effects of compounds on cancer cell proliferation?",
    },
    {
        "label": "Query 2 (precise)",
        "text": "cytotoxicity HeLa cells",
    },
    {
        "label": "Query 3 (hors contexte)",
        "text": "stock market financial results 2024",
    },
]


def run_query(retriever: EnsembleRetriever, label: str, query: str) -> tuple[list[dict], float]:
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"  '{query}'")
    t0 = time.time()
    results = retriever.retrieve(query, top_k=3)
    elapsed = time.time() - t0
    print(f"  Temps : {elapsed:.2f}s | {len(results)} resultats")
    for i, doc in enumerate(results, 1):
        content_preview = doc["content"][:150].replace("\n", " ") + "..."
        print(f"\n  [{i}] rerank_score={doc.get('rerank_score', 0):.4f}")
        print(f"       source_file : {doc['metadata'].get('source_file', 'N/A')}")
        print(f"       content     : {content_preview}")
    return results, elapsed


def main():
    print("=" * 60)
    print("TEST RETRIEVAL — pipeline BM25 + Vector + Reranker")
    print("=" * 60)
    print("\nLoading EnsembleRetriever (BM25 index + Qdrant + Reranker)...")
    retriever = EnsembleRetriever()
    print("Ready.\n")

    all_results = {}
    for q in QUERIES:
        results, elapsed = run_query(retriever, q["label"], q["text"])
        all_results[q["label"]] = results

    # Validation
    print(f"\n{'=' * 60}")
    errors = []
    for q in QUERIES[:2]:  # only first 2 must return results
        docs = all_results[q["label"]]
        if not docs:
            errors.append(f"{q['label']}: no results returned")
            continue
        bad = [d for d in docs if d.get("rerank_score", -1) <= 0.0]
        if bad:
            errors.append(f"{q['label']}: {len(bad)} result(s) with rerank_score <= 0.0")

    if errors:
        for err in errors:
            print(f"FAIL : {err}")
        sys.exit(1)
    else:
        print("PASS : 2 premieres queries ont des resultats avec rerank_score > 0.0")


if __name__ == "__main__":
    main()
