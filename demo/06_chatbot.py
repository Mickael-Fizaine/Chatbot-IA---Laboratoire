"""
DEMO 06 — Chatbot Interactif en Direct
=======================================
Lance un chatbot en ligne de commande.
La reponse est toujours dans la meme langue que la question.

Prerequis : Qdrant en cours + GOOGLE_API_KEY dans .env
Commandes  : /exit  /quit  /help  /clear  /lang
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

W = 74

# Language detection (optional dependency)
try:
    from langdetect import detect as _detect_lang
    def detect_lang(text: str) -> str:
        try:
            return _detect_lang(text)
        except Exception:
            return "?"
except ImportError:
    def detect_lang(text: str) -> str:
        return "?"

LANG_LABELS = {
    "fr": "Francais", "en": "English", "de": "Deutsch",
    "es": "Espanol",  "it": "Italiano", "nl": "Nederlands",
    "?":  "Inconnu",
}

HELP_TEXT = """
  Commandes disponibles :
  ─────────────────────────────────────────
  /exit  ou  /quit   Quitter le chatbot
  /clear             Effacer l'ecran
  /lang              Afficher la langue detectee de votre derniere question
  /help              Afficher ce message
  ─────────────────────────────────────────
  Posez vos questions en francais, anglais ou toute autre langue.
  Le chatbot repond dans la meme langue que votre question.
"""


def _sep(char: str = "─") -> str:
    return char * W


def _box_response(answer: str, sources: list, relevance: float, hallucination: float, elapsed: float) -> None:
    print(f"\n  {'─' * (W - 2)}")
    print(f"  Bot >\n")

    # Wrap answer
    for i in range(0, len(answer), W - 6):
        print(f"    {answer[i:i + W - 6]}")

    if sources:
        print(f"\n    Sources :")
        for s in sources[:5]:
            print(f"      - {s}")

    print(f"\n    Pertinence : {relevance:.2f}  |  Fiabilite : {hallucination:.2f}  |  {elapsed:.0f}s")
    print(f"  {'─' * (W - 2)}\n")


def _welcome() -> None:
    print(f"\n{'═' * W}")
    print(f"{'  LAB CHATBOT — Assistant R&D Pharmaceutique / Cosmetique':^{W}}")
    print(f"{'═' * W}")
    print(f"  Posez vos questions sur les composes testes, les resultats d'etudes,")
    print(f"  les effets sur les cellules, les IC50, etc.")
    print(f"  La reponse sera dans la meme langue que votre question.")
    print(f"  Tapez /help pour les commandes.")
    print(f"{'─' * W}\n")


def run() -> None:
    from graph.graph import process_query

    _welcome()

    last_lang = "?"
    history: list[dict] = []

    while True:
        try:
            raw = input(f"  Vous > ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  Au revoir !\n")
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────────────
        if raw.lower() in ("/exit", "/quit"):
            print(f"\n  Au revoir !\n")
            break

        if raw.lower() == "/clear":
            os.system("cls" if sys.platform == "win32" else "clear")
            _welcome()
            continue

        if raw.lower() == "/help":
            print(HELP_TEXT)
            continue

        if raw.lower() == "/lang":
            label = LANG_LABELS.get(last_lang, last_lang)
            print(f"\n  Derniere langue detectee : {label} ({last_lang})\n")
            continue

        # ── Language detection ─────────────────────────────────────────
        last_lang = detect_lang(raw)
        lang_label = LANG_LABELS.get(last_lang, last_lang)
        print(f"\n  [Langue : {lang_label}]  [Recherche en cours...]", end="", flush=True)

        # ── Query ──────────────────────────────────────────────────────
        t0 = time.time()
        try:
            result = process_query(raw)
            elapsed = time.time() - t0

            # Erase the "Recherche en cours..." line
            print(f"\r{' ' * W}\r", end="")

            answer      = result.get("answer", "Information non disponible.")
            sources     = result.get("sources", [])
            relevance   = result.get("relevance_score", 0.0)
            hallucinate = result.get("hallucination_score", 0.0)
            qtype       = result.get("query_type", "")

            if qtype == "out_of_scope":
                answer = (
                    "Cette question est hors du perimetre de la base de connaissances."
                    if last_lang == "fr"
                    else "This question is outside the scope of the knowledge base."
                )

            _box_response(answer, sources, relevance, hallucinate, elapsed)
            history.append({"q": raw, "a": answer, "lang": last_lang})

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"\r{' ' * W}\r", end="")
            print(f"\n  [ERREUR] {type(exc).__name__}: {exc}  ({elapsed:.0f}s)\n")


if __name__ == "__main__":
    run()
