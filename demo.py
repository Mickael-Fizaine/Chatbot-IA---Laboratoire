import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from graph.graph import process_query

QUESTIONS = [
    {
        "type": "Semantique",
        "query": "What are the observed effects on cell proliferation during efficacy tests?",
        "criteria": "query_type=semantic, sources non vides, hallucination >= 0.50",
    },
    {
        "type": "Precision exacte",
        "query": "What are the cytotoxicity results for 2-methoxyestradiol?",
        "criteria": "query_type=precise, sources non vides",
    },
    {
        "type": "Synthese comparative",
        "query": "Compare the efficacy of compounds tested on cancer cells versus healthy cells",
        "criteria": "query_type=semantic, reponse non vide",
    },
    {
        "type": "Hors scope",
        "query": "What is the current stock price of L'Oreal?",
        "criteria": "query_type=out_of_scope",
    },
    {
        "type": "Piege hallucination",
        "query": "Has compound ABC-9999 ever been tested in your database?",
        "criteria": "reponse contient 'non disponible' OU sources vides",
    },
    {
        "type": "Ambigue",
        "query": "Did we already do this test?",
        "criteria": "pas de crash, reponse retournee",
    },
]

W = 56  # box width


def _sep(char="="):
    return char * W


def _evaluate(result: dict, criteria: str) -> bool:
    qt = result.get("query_type", "")
    sources = result.get("sources", [])
    answer = result.get("answer", "")
    hall = result.get("hallucination_score", 0.0)

    if "query_type=semantic" in criteria and qt not in ("semantic", "precise"):
        return False
    if "query_type=precise" in criteria and qt != "precise":
        return False
    if "query_type=out_of_scope" in criteria and qt != "out_of_scope":
        return False
    if "sources non vides" in criteria and not sources:
        return False
    if "hallucination >= 0.50" in criteria and hall < 0.50:
        return False
    if "reponse non vide" in criteria and not answer.strip():
        return False
    if ("'non disponible' OU sources vides" in criteria
            and "non disponible" not in answer.lower()
            and sources):
        return False
    return True


def _format_response(result: dict) -> str:
    qt = result.get("query_type", "")
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    relevance = result.get("relevance_score", 0.0)
    hallucination = result.get("hallucination_score", 0.0)

    if qt == "out_of_scope":
        return "Hors perimetre de la base de connaissances du laboratoire."

    lines = ["Reponse :", "", answer]
    if sources:
        lines += ["", "Sources :"] + [f"  - {s}" for s in sources[:5]]
    lines += ["", f"Pertinence : {relevance:.2f}  |  Fiabilite : {hallucination:.2f}"]
    return "\n".join(lines)


def run_question(index: int, total: int, item: dict) -> tuple[bool, float]:
    label = item["type"].upper()
    print(f"\n{_sep()}")
    print(f"[{index}/{total}] {label}")
    print(_sep())
    print(f"Question : {item['query']}\n")

    t0 = time.time()
    try:
        result = process_query(item["query"])
        elapsed = time.time() - t0

        formatted = _format_response(result)
        print(formatted)

        qt = result.get("query_type", "")
        rel = result.get("relevance_score", 0.0)
        hall = result.get("hallucination_score", 0.0)
        print(f"\nQuery type    : {qt}")
        print(f"Relevance     : {rel:.2f}")
        print(f"Hallucination : {hall:.2f}")
        print(f"Temps         : {elapsed:.1f}s")

        passed = _evaluate(result, item["criteria"])

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"ERREUR : {exc}")
        print(f"Temps  : {elapsed:.1f}s")
        # "pas de crash" criterion fails on exception for all except ambiguous
        passed = False

    status = "PASS" if passed else "FAIL"
    print(f"\nCritere : {item['criteria']}")
    print(f"Resultat : [{status}]")
    return passed, elapsed


def print_summary(items: list[dict], results: list[tuple[bool, float]]):
    total_time = sum(e for _, e in results)
    passed_n = sum(1 for ok, _ in results if ok)
    total = len(items)

    inner = W - 2  # content width inside box borders

    def row(text: str) -> str:
        return f"|  {text:<{inner - 2}}|"

    print(f"\n+{_sep('-')}+")
    title = "RECAPITULATIF DEMO -- Lab Chatbot"
    print(f"|{title:^{inner}}|")
    print(f"+{_sep('-')}+")

    for i, (item, (ok, elapsed)) in enumerate(zip(items, results), 1):
        status = "PASS" if ok else "FAIL"
        label = f"[{i}] {item['type'][:20]:<20} {status:<6} {elapsed:.1f}s"
        print(row(label))

    print(f"+{_sep('-')}+")
    print(row(f"Score global : {passed_n}/{total}"))
    print(row(f"Temps total  : {total_time:.1f}s"))
    print(f"+{_sep('-')}+")


def main():
    print(_sep())
    print("DEMO -- Lab Chatbot RAG Pipeline")
    print(_sep())
    print(f"{len(QUESTIONS)} questions de test\n")

    all_results: list[tuple[bool, float]] = []
    for i, item in enumerate(QUESTIONS, 1):
        ok, elapsed = run_question(i, len(QUESTIONS), item)
        all_results.append((ok, elapsed))

    print_summary(QUESTIONS, all_results)


if __name__ == "__main__":
    main()
