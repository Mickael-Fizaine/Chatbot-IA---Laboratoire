"""
DEMO 04 — Evaluation des Metriques du Lab Chatbot
===================================================
Calcule sur 10 questions de test :
  - Faithfulness        (proxy : % reponses ancrees dans les sources)
  - Answer Relevancy    (proxy : score de pertinence moyen)
  - Context Precision   (proxy : % requetes avec relevance_score >= 0.45)
  - Context Recall      (proxy : % requetes avec au moins 1 source retournee)
  - Taux "Non disponible"
  - Latence bout-en-bout (min / moy / max / P95)

Si evaluation/ragas_report.json existe, les metriques RAGAS officielles
(LLM-as-judge) sont aussi affichees.

Prerequis : Qdrant en cours + GOOGLE_API_KEY dans .env
Attention  : 10 appels Gemini — peut prendre ~15-30 min sur tier gratuit
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from demo.utils import (
    W, bar, box, header, metric_line, save_json, save_txt,
    section, sep, step_banner, table,
)

NOT_AVAILABLE_MSG = "non disponible"

TEST_SET = [
    # (id, type, query, expected_type)
    ("Q01", "In-scope / precis",      "What are the cytotoxicity results for 2-methoxyestradiol?",                        "precise"),
    ("Q02", "In-scope / semantique",  "What are the observed effects on cell proliferation during efficacy tests?",       "semantic"),
    ("Q03", "In-scope / comparatif",  "Compare the efficacy of compounds tested on cancer cells versus healthy cells",    "semantic"),
    ("Q04", "In-scope / mecanisme",   "What is the mechanism of action of paclitaxel on tumor cell growth?",              "precise"),
    ("Q05", "In-scope / cellulaire",  "Which cell lines showed the highest sensitivity to cytotoxic compounds?",         "semantic"),
    ("Q06", "Hors-scope / general",   "What is the current stock price of L'Oreal?",                                     "out_of_scope"),
    ("Q07", "Hors-scope / meteo",     "What is the weather forecast for Paris this weekend?",                             "out_of_scope"),
    ("Q08", "Compose inconnu",        "Has compound ABC-9999 ever been tested in your database?",                         "semantic"),
    ("Q09", "Compose inconnu",        "What are the side effects of drug XYZ-47821?",                                    "semantic"),
    ("Q10", "Requete ambigue",        "Did we already do this test?",                                                     "semantic"),
]


def run_test_set(out) -> list[dict]:
    from graph.graph import process_query

    results = []
    out(f"\n  {len(TEST_SET)} questions de test — execution en cours...\n")
    out(table(
        ["ID", "Type", "Requete (extrait)"],
        [[q[0], q[1], q[2][:50]] for q in TEST_SET],
        max_col=52,
    ))
    out("")

    for qid, qtype, query, expected in TEST_SET:
        out(f"  [{qid}] {query[:60]}...")
        t0 = time.time()
        try:
            r = process_query(query)
            elapsed = time.time() - t0
            not_available = NOT_AVAILABLE_MSG in r.get("answer", "").lower()
            correct_type  = r["query_type"] == expected
            results.append({
                "id": qid,
                "type": qtype,
                "query": query,
                "expected_type": expected,
                "query_type": r["query_type"],
                "type_correct": correct_type,
                "relevance_score": r.get("relevance_score", 0.0),
                "hallucination_score": r.get("hallucination_score", 0.0),
                "sources": r.get("sources", []),
                "answer": r.get("answer", ""),
                "not_available": not_available,
                "latency_s": round(elapsed, 1),
                "error": None,
            })
            status = "OK" if correct_type else "type?"
            out(f"         -> type={r['query_type']:<12} rel={r['relevance_score']:.2f}  hall={r['hallucination_score']:.2f}  {elapsed:.0f}s  [{status}]")
        except Exception as exc:
            elapsed = time.time() - t0
            out(f"         -> ERREUR ({type(exc).__name__}: {exc})  {elapsed:.0f}s")
            results.append({
                "id": qid, "type": qtype, "query": query,
                "expected_type": expected, "query_type": "", "type_correct": False,
                "relevance_score": 0.0, "hallucination_score": 0.0,
                "sources": [], "answer": "", "not_available": True,
                "latency_s": round(elapsed, 1), "error": str(exc),
            })
    return results


def compute_metrics(results: list[dict]) -> dict:
    in_scope = [r for r in results if r["expected_type"] != "out_of_scope" and not r["error"]]
    oos      = [r for r in results if r["expected_type"] == "out_of_scope"]
    all_ok   = [r for r in results if not r["error"]]

    # Faithfulness : % reponses ancrees (hallucination_score >= 0.50) parmi in-scope
    faithfulness = (
        sum(1 for r in in_scope if r["hallucination_score"] >= 0.50) / len(in_scope)
        if in_scope else 0.0
    )

    # Answer Relevancy : moyenne des scores de pertinence (in-scope uniquement)
    answer_relevancy = (
        sum(r["relevance_score"] for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )

    # Context Precision : % requetes in-scope ou relevance_score >= 0.45
    context_precision = (
        sum(1 for r in in_scope if r["relevance_score"] >= 0.45) / len(in_scope)
        if in_scope else 0.0
    )

    # Context Recall : % requetes in-scope qui ont retourne au moins 1 source
    context_recall = (
        sum(1 for r in in_scope if r["sources"]) / len(in_scope)
        if in_scope else 0.0
    )

    # Taux "non disponible" (in-scope seulement — normal pour out-of-scope)
    not_available_rate = (
        sum(1 for r in in_scope if r["not_available"]) / len(in_scope)
        if in_scope else 0.0
    )

    # Taux de detection hors-scope correct
    oos_accuracy = (
        sum(1 for r in oos if r["query_type"] == "out_of_scope") / len(oos)
        if oos else 0.0
    )

    # Latence
    latencies = [r["latency_s"] for r in all_ok]
    latencies_sorted = sorted(latencies)
    p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)

    return {
        "faithfulness":       round(faithfulness,       3),
        "answer_relevancy":   round(answer_relevancy,   3),
        "context_precision":  round(context_precision,  3),
        "context_recall":     round(context_recall,     3),
        "not_available_rate": round(not_available_rate, 3),
        "oos_accuracy":       round(oos_accuracy,       3),
        "latency": {
            "min_s":  round(min(latencies), 1) if latencies else 0,
            "max_s":  round(max(latencies), 1) if latencies else 0,
            "avg_s":  round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "p95_s":  round(latencies_sorted[p95_idx], 1) if latencies_sorted else 0,
        },
        "n_in_scope":  len(in_scope),
        "n_oos":       len(oos),
        "n_total":     len(results),
    }


def load_ragas_report() -> dict | None:
    path = ROOT / "evaluation" / "ragas_report.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def display_report(metrics: dict, results: list[dict], ragas: dict | None, out):
    out(header(
        "RAPPORT METRIQUES — Lab Chatbot",
        f"Evalue sur {metrics['n_total']} questions  "
        f"({metrics['n_in_scope']} in-scope, {metrics['n_oos']} hors-scope)"
    ))

    out("\n  METRIQUES PROXY (basees sur les scores internes du pipeline)")
    out(f"  {'─' * 66}")
    out(metric_line("Faithfulness        (hall >= 0.50)", metrics["faithfulness"]))
    out(metric_line("Answer Relevancy    (moy relevance)", metrics["answer_relevancy"]))
    out(metric_line("Context Precision   (rel >= 0.45)",  metrics["context_precision"]))
    out(metric_line("Context Recall      (>= 1 source)",  metrics["context_recall"]))

    out(f"\n  COMPORTEMENT DU SYSTEME")
    out(f"  {'─' * 66}")
    out(metric_line("Detection hors-scope",               metrics["oos_accuracy"]))
    out(metric_line("Taux 'Non disponible' (in-scope)",   metrics["not_available_rate"]))

    lat = metrics["latency"]
    out(f"\n  LATENCE BOUT-EN-BOUT")
    out(f"  {'─' * 66}")
    out(table(
        ["Minimum", "Moyenne", "Maximum", "P95"],
        [[f"{lat['min_s']}s", f"{lat['avg_s']}s", f"{lat['max_s']}s", f"{lat['p95_s']}s"]],
    ))

    if ragas:
        m = ragas.get("metrics", {})
        out(f"\n  METRIQUES RAGAS OFFICIELLES  (LLM-as-judge, evaluation/ragas_report.json)")
        out(f"  {'─' * 66}")
        out(metric_line("Faithfulness      (RAGAS)", m.get("faithfulness", 0)))
        out(metric_line("Answer Relevancy  (RAGAS)", m.get("answer_relevancy", 0)))
        out(metric_line("Context Precision (RAGAS)", m.get("context_precision", 0)))
        out(metric_line("Context Recall    (RAGAS)", m.get("context_recall", 0)))
        avg = m.get("average", 0)
        verdict = ragas.get("verdict", "—")
        out(f"\n  Score moyen RAGAS : {avg:.2f}  {bar(avg)}")
        out(f"  Verdict           : {verdict}")
    else:
        out(f"\n  Metriques RAGAS officielles non disponibles.")
        out(f"  Pour les obtenir : python evaluation/ragas_eval.py")

    out(f"\n  DETAIL PAR QUESTION")
    out(f"  {'─' * 66}")
    out(table(
        ["ID", "Type attendu", "Type obtenu", "Rel.", "Hall.", "Latence", "Statut"],
        [
            [
                r["id"],
                r["expected_type"][:16],
                r["query_type"][:12] if r["query_type"] else "ERROR",
                f"{r['relevance_score']:.2f}",
                f"{r['hallucination_score']:.2f}",
                f"{r['latency_s']}s",
                "OK" if r["type_correct"] else ("ERR" if r["error"] else "?type"),
            ]
            for r in results
        ],
    ))


def run():
    out_lines: list[str] = []

    def p(text: str = ""):
        print(text)
        out_lines.append(text)

    p(header(
        "DEMO 04 — Evaluation des Metriques",
        "Faithfulness | Relevancy | Precision | Recall | Latence"
    ))

    p(step_banner(1, 3, "Execution du jeu de test (10 questions)"))
    results = run_test_set(p)

    p(step_banner(2, 3, "Calcul des Metriques"))
    metrics = compute_metrics(results)

    p(step_banner(3, 3, "Rapport Final"))
    ragas = load_ragas_report()
    display_report(metrics, results, ragas, p)

    p(f"\n{sep()}")
    p("* Proxy = calcule a partir des scores internes (relevance_score et")
    p("  hallucination_score issus de gemini-2.5-flash). Pour les metriques")
    p("  RAGAS completes avec LLM-as-judge, lancer : python evaluation/ragas_eval.py")
    p(sep())

    full_output = {
        "metrics": metrics,
        "per_question": results,
        "ragas_available": ragas is not None,
    }
    save_json("04_metrics", full_output)
    save_txt("04_metrics", "\n".join(out_lines))


if __name__ == "__main__":
    run()
