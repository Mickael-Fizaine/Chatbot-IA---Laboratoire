import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.nodes.state import GraphState

load_dotenv()

_SYSTEM_PROMPT = """Tu es un vérificateur de fidélité scientifique.
Compare la réponse générée avec les documents sources fournis.
Retourne UNIQUEMENT un JSON valide sans markdown :
{"hallucination_score": float entre 0 et 1, "is_grounded": bool, "issues": "description si problème"}

Critères :
- 1.0 : chaque affirmation est traceable dans les sources
- 0.5-1.0 : majorité des affirmations traceable
- 0.0-0.5 : affirmations inventées ou non traceable → hallucination détectée
"""

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)


def node_hallucination_checker(state: GraphState) -> GraphState:
    answer = state.get("answer", "")
    documents = state.get("documents", [])

    context_parts = []
    for i, doc in enumerate(documents[:5], 1):
        snippet = doc.get("content", "")[:400]
        source = doc.get("metadata", {}).get("source_file", f"doc_{i}")
        context_parts.append(f"[{source}]: {snippet}")
    context = "\n".join(context_parts) or "Aucun document."

    try:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Réponse générée :\n{answer}\n\nDocuments sources :\n{context}"
            ),
        ]
        response = _llm.invoke(messages)
        parsed = _parse_json(response.content)
        score = float(parsed.get("hallucination_score", 0.75))
        state["hallucination_score"] = score
        if score < 0.50:
            state["answer"] = "[REGENERATED] " + answer
    except Exception:
        state["hallucination_score"] = 0.75
    return state
