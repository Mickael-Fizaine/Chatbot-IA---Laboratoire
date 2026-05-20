import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from graph.nodes.state import GraphState
from graph.nodes.query_analyzer import node_query_analyzer
from graph.nodes.retriever import node_retriever
from graph.nodes.relevance_grader import node_relevance_grader
from graph.nodes.generator import node_generator
from graph.nodes.hallucination_checker import node_hallucination_checker

load_dotenv()

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

workflow = StateGraph(GraphState)

workflow.add_node("query_analyzer", node_query_analyzer)
workflow.add_node("retriever", node_retriever)
workflow.add_node("relevance_grader", node_relevance_grader)
workflow.add_node("generator", node_generator)
workflow.add_node("hallucination_checker", node_hallucination_checker)

workflow.set_entry_point("query_analyzer")


def route_after_analyzer(state: GraphState) -> str:
    if state["query_type"] == "out_of_scope":
        return "end_out_of_scope"
    return "retriever"


workflow.add_conditional_edges(
    "query_analyzer",
    route_after_analyzer,
    {"end_out_of_scope": END, "retriever": "retriever"},
)

workflow.add_edge("retriever", "relevance_grader")


def route_after_grader(state: GraphState) -> str:
    if state["relevance_score"] >= 0.45:
        return "generator"
    if state["reformulation_count"] < 2:
        return "retriever"
    return "end_not_found"


workflow.add_conditional_edges(
    "relevance_grader",
    route_after_grader,
    {"generator": "generator", "retriever": "retriever", "end_not_found": END},
)

workflow.add_edge("generator", "hallucination_checker")


def route_after_hallucination(state: GraphState) -> str:
    if state["hallucination_score"] >= 0.50:
        return "end_success"
    if state.get("regeneration_count", 0) >= 2:
        return "end_low_confidence"
    return "generator"


workflow.add_conditional_edges(
    "hallucination_checker",
    route_after_hallucination,
    {"end_success": END, "end_low_confidence": END, "generator": "generator"},
)

app = workflow.compile()

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_MSG = "Cette question est hors du périmètre du laboratoire pharmaceutique/cosmétique."
_NOT_FOUND_MSG = "Information non disponible dans la base de connaissances."


def process_query(query: str) -> dict:
    initial_state: GraphState = {
        "query": query,
        "rewritten_query": "",
        "query_type": "",
        "metadata_filters": {},
        "documents": [],
        "relevance_score": 0.0,
        "answer": "",
        "sources": [],
        "hallucination_score": 0.0,
        "reformulation_count": 0,
        "regeneration_count": 0,
        "final_answer": "",
        "error": "",
    }

    final_state = app.invoke(initial_state)

    # Resolve the final answer depending on exit path
    if final_state["query_type"] == "out_of_scope":
        answer = _OUT_OF_SCOPE_MSG
    elif not final_state.get("answer"):
        answer = _NOT_FOUND_MSG
    else:
        answer = final_state.get("final_answer") or final_state["answer"]

    return {
        "answer": answer,
        "sources": final_state.get("sources", []),
        "query_type": final_state["query_type"],
        "relevance_score": final_state.get("relevance_score", 0.0),
        "hallucination_score": final_state.get("hallucination_score", 0.0),
    }
