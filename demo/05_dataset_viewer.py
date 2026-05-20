"""
DEMO 05 — Lecteur de Documents Scientifiques
=============================================
Affiche le contenu integral de documents du jeu de donnees
pour visualiser de quoi parlent vraiment les donnees.

Prerequis : data/documents.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from demo.utils import W, box, header, sep, table

DOCS_PATH = ROOT / "data" / "documents.json"


def _parse_sections(content: str) -> dict[str, str]:
    """Try to split PubMedQA content into Background / Question / Answer."""
    sections = {"background": "", "question": "", "answer": ""}
    q_idx = content.find("Question:")
    a_idx = content.find("Answer:")
    if q_idx != -1 and a_idx != -1:
        sections["background"] = content[:q_idx].strip()
        sections["question"]   = content[q_idx + 9:a_idx].strip()
        sections["answer"]     = content[a_idx + 7:].strip()
    else:
        sections["background"] = content
    return sections


def _word_count(text: str) -> int:
    return len(text.split())


def display_document(doc: dict, index: int, total: int) -> None:
    m = doc["metadata"]
    content = doc["content"]
    sections = _parse_sections(content)

    print(header(
        f"DOCUMENT {index}/{total}  —  {m['source_file']}",
        f"Domaine : {m['domain']}  |  Langue : {m['language']}  |  Confiance : {m['confidence_extraction']}",
    ))

    # Metadata table
    print(table(
        ["Champ", "Valeur"],
        [
            ["source_file",  m["source_file"]],
            ["file_type",    m["file_type"]],
            ["compound",     m["compound_name"]],
            ["cell_type",    m["cell_type"]],
            ["chunk_index",  str(m["chunk_index"])],
            ["page_number",  str(m["page_number"])],
            ["chunk_type",   m["chunk_type"]],
        ],
    ))

    print(f"\n{sep()}")

    if sections["background"]:
        print("\n  [ CONTEXTE SCIENTIFIQUE ]\n")
        # Print full background, wrapped at W-4 chars
        bg = sections["background"]
        for i in range(0, len(bg), W - 4):
            print(f"  {bg[i:i + W - 4]}")

    if sections["question"]:
        print(f"\n{sep('─')}")
        print(f"\n  [ QUESTION DE RECHERCHE ]\n")
        q = sections["question"]
        for i in range(0, len(q), W - 4):
            print(f"  {q[i:i + W - 4]}")

    if sections["answer"]:
        print(f"\n{sep('─')}")
        print(f"\n  [ CONCLUSION / REPONSE ]\n")
        a = sections["answer"]
        for i in range(0, len(a), W - 4):
            print(f"  {a[i:i + W - 4]}")

    words = _word_count(content)
    chars = len(content)
    print(f"\n{sep()}")
    print(f"  Longueur : {chars:,} caracteres  |  Mots : {words:,}")
    print(sep())


def pick_interesting_docs(docs: list[dict], n: int = 3) -> list[dict]:
    """Pick n docs that have known compound AND cell type for readability."""
    interesting = [
        d for d in docs
        if d["metadata"]["compound_name"] != "unknown"
        and d["metadata"]["cell_type"] != "unknown"
    ]
    step = max(1, len(interesting) // n)
    picks = [interesting[i * step] for i in range(n) if i * step < len(interesting)]
    # Pad with any doc if not enough
    i = 0
    while len(picks) < n and i < len(docs):
        if docs[i] not in picks:
            picks.append(docs[i])
        i += 1
    return picks[:n]


def run():
    if not DOCS_PATH.exists():
        print(f"[ERREUR] {DOCS_PATH} introuvable.")
        print("  -> Lancez d'abord : python data/load_large_dataset.py")
        sys.exit(1)

    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    samples = pick_interesting_docs(docs, n=3)

    print(header(
        "DEMO 05 — Lecteur de Documents Scientifiques",
        f"{len(docs)} documents disponibles — affichage de 3 exemples complets",
    ))
    print(f"\n  Ce demo montre le contenu INTEGRAL de documents du jeu de donnees.")
    print(f"  Chaque document est une publication scientifique biomédicale réelle.\n")

    for i, doc in enumerate(samples, 1):
        input(f"\n  [ Appuyez sur ENTREE pour afficher le document {i}/{len(samples)} ]")
        display_document(doc, i, len(samples))

    print(f"\n  Fin de la lecture. {len(samples)} documents affiches sur {len(docs)} disponibles.")
    print(f"  Dataset actif : data/documents.json  ({len(docs):,} documents)")


if __name__ == "__main__":
    run()
