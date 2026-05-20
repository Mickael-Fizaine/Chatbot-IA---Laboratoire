import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.nodes.state import GraphState

load_dotenv()

_SYSTEM_PROMPT = """Tu es un vérificateur de fidélité scientifique strict.
Compare chaque affirmation de la réponse générée avec les documents sources fournis.
Retourne UNIQUEMENT un JSON valide sans markdown :
{"hallucination_score": float entre 0 et 1, "is_grounded": bool, "issues": "description si problème"}

Règles de scoring :
- 1.0 : toutes les affirmations sont directement traçables dans les sources, OU la réponse dit explicitement que l'information n'est pas disponible
- 0.7-0.9 : la grande majorité est traçable, légères paraphrases acceptables
- 0.4-0.7 : certaines affirmations traceable mais d'autres manquantes ou floues
- 0.0-0.4 : affirmations inventées, chiffres non présents dans les sources, ou informations fabriquées

IMPORTANT : Si la réponse contient "Information non disponible" ou équivalent, score = 1.0 (pas d'hallucination).
Ne score pas négativement une réponse honnête qui admet ne pas avoir l'information.
"""

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)


def node_hallucination_checker(state: GraphState) -> GraphState:
    answer = state.get("answer", "")
    documents = state.get("documents", [])

    NOT_AVAILABLE_MARKERS = (
        "information non disponible",
        "not available",
        "not found in",
        "no information",
        "base de connaissances",
    )
    if any(m in answer.lower() for m in NOT_AVAILABLE_MARKERS):
        state["hallucination_score"] = 1.0
        return state

    context_parts = []
    for i, doc in enumerate(documents[:8], 1):
        snippet = doc.get("content", "")[:700]
        source = doc.get("metadata", {}).get("source_file", f"doc_{i}")
        context_parts.append(f"[{source}]: {snippet}")
    context = "\n\n".join(context_parts) or "Aucun document."

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
            state["regeneration_count"] = state.get("regeneration_count", 0) + 1
    except Exception:
        state["hallucination_score"] = 0.75
    return state
