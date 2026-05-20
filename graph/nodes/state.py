from typing import TypedDict, List, Dict, Any


class GraphState(TypedDict):
    query: str
    rewritten_query: str
    query_type: str
    metadata_filters: dict
    documents: List[Dict[str, Any]]
    relevance_score: float
    answer: str
    sources: List[str]
    hallucination_score: float
    reformulation_count: int  # how many times retrieval was retried
    regeneration_count: int   # how many times generation was retried after low hallucination score
    final_answer: str
    error: str
