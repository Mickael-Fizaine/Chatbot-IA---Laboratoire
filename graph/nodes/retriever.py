from __future__ import annotations
from graph.nodes.state import GraphState

_ensemble: object = None


def _get_retriever():
    global _ensemble
    if _ensemble is None:
        from retrieval.ensemble_retriever import EnsembleRetriever
        _ensemble = EnsembleRetriever()
    return _ensemble


def node_retriever(state: GraphState) -> GraphState:
    # Increment counter on retry calls (documents already populated from a prior pass)
    if state.get("documents"):
        state["reformulation_count"] = state.get("reformulation_count", 0) + 1

    retriever = _get_retriever()
    query = state.get("rewritten_query") or state.get("query", "")
    filters = state.get("metadata_filters") or None

    try:
        documents = retriever.retrieve(query=query, top_k=5, filters=filters)
        state["documents"] = documents
    except Exception as exc:
        state["documents"] = []
        state["error"] = f"Retriever error: {exc}"
    return state
