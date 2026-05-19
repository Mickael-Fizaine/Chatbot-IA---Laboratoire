import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from graph.graph import process_query

QUESTIONS = [
    "What are the effects of 2-methoxyestradiol on tumor growth?",
    "What is the weather in Paris?",
    "Has compound ABC-9999 ever been tested?",
]


def format_bot_response(result: dict) -> str:
    """Replicate the Teams bot formatting logic from LabChatBot."""
    answer = result.get("answer", "Information non disponible.")
    sources = result.get("sources", [])
    query_type = result.get("query_type", "unknown")
    relevance = result.get("relevance_score", 0)
    hallucination = result.get("hallucination_score", 0)

    if query_type == "out_of_scope":
        return "Cette question est hors du perimetre de la base de connaissances du laboratoire."

    response = f"**Reponse**\n\n{answer}"
    if sources:
        sources_text = "\n".join([f"- {s}" for s in sources[:5]])
        response += f"\n\n**Sources**\n{sources_text}"
    response += f"\n\nPertinence : {relevance:.2f} | Fiabilite : {hallucination:.2f}"
    return response


def run_question(question: str) -> bool:
    print(f"\n{'=' * 65}")
    print(f"Question : {question}")
    print("-" * 65)

    try:
        result = process_query(question)
        response = format_bot_response(result)

        print(f"query_type  : {result['query_type']}")
        print(f"relevance   : {result['relevance_score']:.3f}")
        print(f"hallucination: {result['hallucination_score']:.3f}")
        print(f"\nReponse formatee :\n{response}")

        # PASS criteria: response is non-empty and no unhandled exception
        passed = bool(response.strip())
        print(f"\n{'[OK]' if passed else '[FAIL]'} reponse non vide")
        return passed

    except Exception as exc:
        print(f"[FAIL] Exception : {exc}")
        return False


def main():
    print("=" * 65)
    print("TEST BOT — simulation formatage Teams (sans infra Microsoft)")
    print("=" * 65)

    results = [run_question(q) for q in QUESTIONS]

    print(f"\n{'=' * 65}")
    passed_n = sum(results)
    total = len(results)
    if all(results):
        print(f"PASS : {passed_n}/{total} questions traitees correctement")
    else:
        failed = [QUESTIONS[i] for i, ok in enumerate(results) if not ok]
        print(f"FAIL : {passed_n}/{total} OK")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
