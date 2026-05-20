import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.nodes.state import GraphState

load_dotenv()

_SYSTEM_PROMPT = """Tu es un assistant scientifique expert pour un laboratoire spécialisé en tests d'efficacité
de composés cosmétiques et pharmaceutiques.

RÈGLES ABSOLUES :
1. Réponds UNIQUEMENT en te basant sur les documents fournis
2. Si l'information n'est pas dans les documents, réponds exactement :
   "Information non disponible dans la base de connaissances."
3. Cite toujours tes sources entre parenthèses (nom du fichier)
4. Sois précis, factuel et concis
5. Ne jamais inventer de données, résultats ou références
6. Réponds TOUJOURS dans la même langue que la question posée (français → français, English → English, etc.)
"""

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def node_generator(state: GraphState) -> GraphState:
    query = state.get("rewritten_query") or state.get("query", "")
    documents = state.get("documents", [])

    context_parts = []
    sources = []
    for i, doc in enumerate(documents, 1):
        content = doc.get("content", "")
        source = doc.get("metadata", {}).get("source_file", f"doc_{i}")
        context_parts.append(f"[{source}]\n{content}")
        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts) or "Aucun document disponible."

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Documents :\n{context}\n\nQuestion : {query}"),
    ]
    for attempt in range(3):
        try:
            response = _llm.invoke(messages)
            state["answer"] = response.content
            break
        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            if attempt == 2 or not is_rate_limit:
                state["answer"] = "Information non disponible dans la base de connaissances."
                state["error"] = str(exc)
                break
            wait = 60 * (attempt + 1)
            time.sleep(wait)

    state["sources"] = sources
    return state
