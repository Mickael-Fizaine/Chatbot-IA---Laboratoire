import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.nodes.state import GraphState

load_dotenv()

_SYSTEM_PROMPT = """Tu es un analyseur de requêtes pour un chatbot de laboratoire pharmaceutique et cosmétique.
Analyse la question et retourne UNIQUEMENT un JSON valide sans markdown avec :
- query_type : "semantic" si question générale sur des effets/résultats,
               "precise" si la question contient un nom de composé, une référence ou un terme exact,
               "out_of_scope" si la question n'a aucun rapport avec un laboratoire pharma/cosmétique
- rewritten_query : reformulation optimisée pour la recherche documentaire, plus de mots-clés
- metadata_filters : dict avec les clés compound_name, year, test_type si détectés dans la question, sinon {}

Exemples :
Question: "What are HeLa cell cytotoxicity results?"
Réponse: {"query_type": "precise", "rewritten_query": "HeLa cells cytotoxicity test results cancer", "metadata_filters": {"cell_type": "HeLa"}}

Question: "What is the weather today?"
Réponse: {"query_type": "out_of_scope", "rewritten_query": "", "metadata_filters": {}}
"""

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def _parse_json(text: str) -> dict:
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)


def node_query_analyzer(state: GraphState) -> GraphState:
    query = state.get("query", "")
    try:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {query}"),
        ]
        response = _llm.invoke(messages)
        parsed = _parse_json(response.content)
        state["query_type"] = parsed.get("query_type", "semantic")
        state["rewritten_query"] = parsed.get("rewritten_query", query)
        state["metadata_filters"] = parsed.get("metadata_filters", {})
    except Exception:
        state["query_type"] = "semantic"
        state["rewritten_query"] = query
        state["metadata_filters"] = {}
    return state
