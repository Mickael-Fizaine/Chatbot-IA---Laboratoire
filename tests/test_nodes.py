import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.nodes.state import GraphState
from graph.nodes.query_analyzer import node_query_analyzer

QUERIES = [
    {
        "label": "Semantique",
        "query": "What are the effects of compounds on cell proliferation?",
        "expected_type": "semantic",
    },
    {
        "label": "Precise",
        "query": "What are HeLa cell cytotoxicity results for doxorubicin?",
        "expected_type": "precise",
    },
    {
        "label": "Hors scope",
        "query": "What is the weather forecast for Paris tomorrow?",
        "expected_type": "out_of_scope",
    },
]


def make_state(query: str) -> GraphState:
    return GraphState(
        query=query,
        rewritten_query="",
        query_type="",
        metadata_filters={},
        documents=[],
        relevance_score=0.0,
        answer="",
        sources=[],
        hallucination_score=0.0,
        reformulation_count=0,
        final_answer="",
        error="",
    )


def main():
    print("=" * 60)
    print("TEST NODES — query_analyzer")
    print("=" * 60)

    errors = []

    for q in QUERIES:
        state = make_state(q["query"])
        result = node_query_analyzer(state)

        qt = result["query_type"]
        rq = result["rewritten_query"]
        mf = result["metadata_filters"]
        match = qt == q["expected_type"]

        status = "OK" if match else "FAIL"
        print(f"\n[{status}] {q['label']}")
        print(f"  query          : {q['query']}")
        print(f"  query_type     : {qt}  (expected: {q['expected_type']})")
        print(f"  rewritten      : {rq}")
        print(f"  metadata_filters: {mf}")

        if not match:
            errors.append(f"{q['label']}: expected {q['expected_type']!r}, got {qt!r}")

    print(f"\n{'=' * 60}")
    if errors:
        for err in errors:
            print(f"FAIL : {err}")
        sys.exit(1)
    else:
        print("PASS : query_analyzer classifie correctement les 3 types de requetes")


if __name__ == "__main__":
    main()
