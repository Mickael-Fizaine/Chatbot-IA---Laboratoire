import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.nodes.state import GraphState

load_dotenv()

try:
    from langdetect import detect as _detect
    def _lang(text: str) -> str:
        try:
            return _detect(text)
        except Exception:
            return "fr"
except ImportError:
    def _lang(text: str) -> str:
        return "fr"

_LANG_NAMES = {
    "fr": "French", "en": "English", "de": "German",
    "es": "Spanish", "it": "Italian", "nl": "Dutch",
    "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese",
    "ar": "Arabic",
}

_SYSTEM_PROMPT = """Tu es un assistant scientifique expert pour un laboratoire spécialisé en tests d'efficacité
de composés cosmétiques et pharmaceutiques.

RÈGLES ABSOLUES :
1. Réponds UNIQUEMENT en te basant sur les documents fournis entre crochets [source].
2. Pour chaque valeur numérique, IC50, pourcentage ou résultat que tu mentionnes, celle-ci DOIT apparaître
   littéralement dans les documents fournis. Si tu ne la trouves pas mot pour mot, ne la cite pas.
3. Si la question porte sur un composé, un médicament ou une donnée ABSENTE des documents, réponds EXACTEMENT :
   "Information non disponible dans la base de connaissances."
4. Cite toujours tes sources entre parenthèses (nom du fichier source).
5. Ne jamais interpoler, extrapoler, ni compléter avec tes connaissances générales.
6. Réponds TOUJOURS dans la même langue que la question posée.
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

    detected = _lang(query)
    lang_name = _LANG_NAMES.get(detected, detected.upper())

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Documents :\n{context}\n\n"
            f"Question : {query}\n\n"
            f"IMPORTANT: You MUST write your entire answer in {lang_name}. "
            f"Do not use any other language."
        )),
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
