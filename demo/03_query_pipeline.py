"""
DEMO 03 — Parcours d'une Requete dans le Pipeline
===================================================
Montre chaque etape : Analyse → BM25 → Vectoriel → Fusion → Rerank → LangGraph

Prerequis : Qdrant en cours + GOOGLE_API_KEY dans .env
Attention  : effectue de vrais appels Gemini (quota consomme)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from demo.utils import (
    W, bar, box, header, save_txt, section, sep, step_banner, table
)

DEMO_QUERY = "What are the cytotoxicity results for 2-methoxyestradiol?"


# ---------------------------------------------------------------------------
# Step 1 — Query analysis
# ---------------------------------------------------------------------------

def step_query_analysis(query: str, out) -> dict:
    out(f"\n  Requete originale : \"{query}\"\n")
    from graph.nodes.query_analyzer import node_query_analyzer
    from graph.nodes.state import GraphState

    dummy: GraphState = {
        "query": query, "rewritten_query": "", "query_type": "",
        "metadata_filters": {}, "documents": [], "relevance_score": 0.0,
        "answer": "", "sources": [], "hallucination_score": 0.0,
        "reformulation_count": 0, "regeneration_count": 0,
        "final_answer": "", "error": "",
    }
    t0 = time.time()
    result = node_query_analyzer(dummy)
    elapsed = time.time() - t0

    out(box([
        f"query_type        : {result['query_type']}",
        f"rewritten_query   : {result['rewritten_query']}",
        f"metadata_filters  : {result['metadata_filters']}",
        sep("─"),
        f"Interpretation    : {elapsed:.1f}s  (appel Gemini gemini-2.5-flash)",
    ]))
    return result


# ---------------------------------------------------------------------------
# Step 2 — BM25 retrieval
# ---------------------------------------------------------------------------

def step_bm25(query: str, out) -> list[dict]:
    from retrieval.bm25_retriever import BM25Retriever
    out(f"\n  Tokenisation BM25 : {query.split()}\n")
    t0 = time.time()
    retriever = BM25Retriever()
    bm25_docs = retriever.retrieve(query, k=20)
    elapsed = time.time() - t0

    top5 = bm25_docs[:5]
    rows = [
        [f"#{i+1}", d["metadata"].get("source_file", "?"), f"{d['score']:.3f}"]
        for i, d in enumerate(top5)
    ]
    out(table(["Rang", "Source", "Score BM25"], rows))
    out(f"\n  {len(bm25_docs)} docs recuperes en {elapsed:.2f}s (index en memoire, pas de reseau)")
    return bm25_docs


# ---------------------------------------------------------------------------
# Step 3 — Vector retrieval
# ---------------------------------------------------------------------------

def step_vector(query: str, out) -> list[dict]:
    from retrieval.vector_retriever import VectorRetriever
    out(f"\n  Embedding de la requete via gemini-embedding-001 (3072 dim)...\n")
    t0 = time.time()
    retriever = VectorRetriever()
    vec_docs = retriever.retrieve(query, k=20)
    elapsed = time.time() - t0

    top5 = vec_docs[:5]
    rows = [
        [f"#{i+1}", d["metadata"].get("source_file", "?"), f"{d['score']:.4f}"]
        for i, d in enumerate(top5)
    ]
    out(table(["Rang", "Source", "Cosine sim."], rows))
    out(f"\n  {len(vec_docs)} docs recuperes en {elapsed:.2f}s (Qdrant HNSW index)")
    return vec_docs


# ---------------------------------------------------------------------------
# Step 4 — Fusion + rerank
# ---------------------------------------------------------------------------

def step_fusion_rerank(query: str, bm25_docs: list, vec_docs: list, out) -> list[dict]:
    from retrieval.reranker import Reranker

    # Reproduce fusion logic from EnsembleRetriever
    seen: dict[str, dict] = {}
    for doc in bm25_docs + vec_docs:
        key = doc["content"]
        if key not in seen:
            seen[key] = dict(doc)
            seen[key]["bm25_score"]   = doc["score"] if doc["source"] == "bm25"   else 0.0
            seen[key]["vector_score"] = doc["score"] if doc["source"] == "vector" else 0.0
        else:
            if doc["source"] == "bm25":
                seen[key]["bm25_score"]   = max(seen[key].get("bm25_score",   0.0), doc["score"])
            else:
                seen[key]["vector_score"] = max(seen[key].get("vector_score", 0.0), doc["score"])

    merged = list(seen.values())

    def norm(vals):
        mn, mx = min(vals), max(vals)
        span = mx - mn
        return [1.0] * len(vals) if span == 0 else [(v - mn) / span for v in vals]

    nb = norm([d["bm25_score"]   for d in merged])
    nv = norm([d["vector_score"] for d in merged])
    for doc, b, v in zip(merged, nb, nv):
        doc["fused_score"] = 0.6 * v + 0.4 * b

    top20 = sorted(merged, key=lambda d: d["fused_score"], reverse=True)[:20]

    out(f"\n  {len(bm25_docs)} docs BM25 + {len(vec_docs)} docs vectoriels")
    out(f"  Apres deduplication : {len(merged)} docs uniques")
    out(f"  Formule fusion      : fused = 0.6 * vector_norm + 0.4 * bm25_norm\n")

    # Rerank
    t0 = time.time()
    reranker = Reranker()
    reranked = reranker.rerank(query, top20, top_n=5)
    elapsed  = time.time() - t0

    rows = [
        [
            f"#{i+1}",
            d["metadata"].get("source_file", "?"),
            f"{d.get('fused_score', 0):.3f}",
            f"{d.get('rerank_score', 0):.4f}",
        ]
        for i, d in enumerate(reranked)
    ]
    out(table(["Rang", "Source", "Fused", "TF-IDF Rerank"], rows))
    out(f"\n  Reranking TF-IDF cosine termine en {elapsed:.3f}s")
    return reranked


# ---------------------------------------------------------------------------
# Step 5 — LangGraph flow diagram
# ---------------------------------------------------------------------------

def show_langgraph_diagram(result: dict, out):
    qt = result.get("query_type", "?")
    rel = result.get("relevance_score", 0.0)
    hall = result.get("hallucination_score", 0.0)

    graded   = "PERTINENT" if rel   >= 0.45 else "NON PERTINENT"
    grounded = "FIABLE"    if hall  >= 0.50 else "FAIBLE FIABILITE"

    out(f"""
  Flux LangGraph traverse :

  ┌──────────────────┐
  │  Query Analyzer  │  type={qt}
  └────────┬─────────┘
           │
           v
  ┌──────────────────┐
  │    Retriever     │  {len(result.get('sources', []))} sources recuperees
  └────────┬─────────┘
           │
           v
  ┌──────────────────┐
  │ Relevance Grader │  score={rel:.2f}  →  {graded}
  └────────┬─────────┘
           │ (>= 0.45)
           v
  ┌──────────────────┐
  │    Generator     │  reponse generee (gemini-2.5-flash)
  └────────┬─────────┘
           │
           v
  ┌──────────────────┐
  │  Halluc. Checker │  score={hall:.2f}  →  {grounded}
  └────────┬─────────┘
           │ (>= 0.50)
           v
       [ FIN OK ]
""")


# ---------------------------------------------------------------------------
# Step 6 — Full pipeline + final answer
# ---------------------------------------------------------------------------

def step_full_pipeline(query: str, out) -> dict:
    from graph.graph import process_query
    out(f"\n  Execution complete du graph LangGraph...\n")
    t0 = time.time()
    result = process_query(query)
    elapsed = time.time() - t0

    out(box([
        f"query_type         : {result['query_type']}",
        f"relevance_score    : {result['relevance_score']:.2f}  {bar(result['relevance_score'])}",
        f"hallucination_score: {result['hallucination_score']:.2f}  {bar(result['hallucination_score'])}",
        sep("─"),
        "Reponse :",
        *[result["answer"][i:i+68] for i in range(0, min(len(result["answer"]), 340), 68)],
        sep("─"),
        "Sources :",
        *[f"  - {s}" for s in result.get("sources", [])[:5]],
        sep("─"),
        f"Latence bout-en-bout : {elapsed:.1f}s",
    ]))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    out_lines: list[str] = []

    def p(text: str = ""):
        print(text)
        out_lines.append(text)

    p(header(
        "DEMO 03 — Parcours d'une Requete dans le Pipeline",
        f'Requete : "{DEMO_QUERY}"',
    ))

    p(step_banner(1, 6, "Analyse de la Requete (LangGraph: node_query_analyzer)"))
    analyzer_result = step_query_analysis(DEMO_QUERY, p)

    p(step_banner(2, 6, "Retrieval BM25  (rank-bm25, index en memoire)"))
    bm25_docs = step_bm25(DEMO_QUERY, p)

    p(step_banner(3, 6, "Retrieval Vectoriel  (Qdrant + gemini-embedding-001)"))
    vec_docs = step_vector(DEMO_QUERY, p)

    p(step_banner(4, 6, "Fusion des Sources + Reranking TF-IDF"))
    reranked = step_fusion_rerank(DEMO_QUERY, bm25_docs, vec_docs, p)

    p(step_banner(5, 6, "Flux LangGraph — Noeuds traverses"))
    result = step_full_pipeline(DEMO_QUERY, p)
    show_langgraph_diagram(result, p)

    p(step_banner(6, 6, "Reponse Finale"))
    p(box([
        f"Requete    : {DEMO_QUERY}",
        sep("─"),
        "Reponse    :",
        *[result["answer"][i:i+68] for i in range(0, min(len(result["answer"]), 500), 68)],
        sep("─"),
        *[f"Source  : {s}" for s in result.get("sources", [])[:5]],
        sep("─"),
        f"Pertinence : {result['relevance_score']:.2f}  |  Fiabilite : {result['hallucination_score']:.2f}",
    ]))

    p(f"\n{sep()}")
    p("CONCLUSION : En une seule requete, le pipeline analyse l'intention,")
    p("fusionne BM25 + vectoriel, reranke par TF-IDF cosine et genere une")
    p("reponse ancree dans les sources — avec verification anti-hallucination.")
    p(sep())

    save_txt("03_query_pipeline", "\n".join(out_lines))


if __name__ == "__main__":
    run()
