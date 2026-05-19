import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.nodes.state import GraphState

load_dotenv()

_SYSTEM_PROMPT = """Tu es un évaluateur de pertinence pour un laboratoire de recherche.
Évalue si les documents fournis permettent de répondre à la question posée.
Retourne UNIQUEMENT un JSON valide sans markdown :
{"relevance_score": float entre 0 et 1, "reason": "explication courte"}

Critères :
- 0.8-1.0 : documents directement pertinents, réponse possible
- 0.5-0.8 : documents partiellement pertinents
- 0.0-0.5 : documents insuffisants ou hors sujet
"""

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)


def node_relevance_grader(state: GraphState) -> GraphState:
    query = state.get("rewritten_query") or state.get("query", "")
    documents = state.get("documents", [])

    context_parts = []
    for i, doc in enumerate(documents[:5], 1):
        snippet = doc.get("content", "")[:300]
        context_parts.append(f"[Doc {i}]: {snippet}")
    context = "\n".join(context_parts) or "Aucun document."

    try:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Question : {query}\n\nDocuments :\n{context}"),
        ]
        response = _llm.invoke(messages)
        parsed = _parse_json(response.content)
        state["relevance_score"] = float(parsed.get("relevance_score", 0.5))
    except Exception:
        state["relevance_score"] = 0.5
    return state
