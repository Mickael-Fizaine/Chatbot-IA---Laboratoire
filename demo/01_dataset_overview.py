"""
DEMO 01 — Apercu du Jeu de Donnees
===================================
Montre : volume, types de supports, richesse scientifique,
exemples de documents et distribution des metadonnees.

Prerequis : data/documents.json doit exister
            (python data/load_pubmedqa.py  ou  python ingestion/parser.py)
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from demo.utils import (
    W, bar, box, header, metric_line, save_txt, section, sep, step_banner, table
)

DOCS_PATH = ROOT / "data" / "documents.json"


def load_docs() -> list[dict]:
    if not DOCS_PATH.exists():
        print(f"[ERREUR] {DOCS_PATH} introuvable.")
        print("  -> Lancez d'abord : python data/load_large_dataset.py")
        sys.exit(1)
    with open(DOCS_PATH, encoding="utf-8") as f:
        return json.load(f)


def run():
    out_lines: list[str] = []

    def p(text: str = ""):
        print(text)
        out_lines.append(text)

    docs = load_docs()
    metas = [d["metadata"] for d in docs]

    p(header(
        "DEMO 01 — Apercu du Jeu de Donnees Lab Chatbot",
        f"{len(docs)} documents charges depuis data/documents.json"
    ))

    # ------------------------------------------------------------------
    # 1 — Volume et types de fichiers
    # ------------------------------------------------------------------
    p(step_banner(1, 5, "Volume & Types de Supports"))
    types = Counter(m["file_type"] for m in metas)
    rows = [[ft, cnt, f"{cnt/len(docs)*100:.1f}%"] for ft, cnt in types.most_common()]
    p(table(["Type de fichier", "Nb documents", "Part"], rows))

    avg_len = sum(len(d["content"]) for d in docs) / len(docs)
    p(f"\n  Total documents   : {len(docs)}")
    p(f"  Longueur moyenne  : {avg_len:,.0f} caracteres / document")

    # ------------------------------------------------------------------
    # 2 — Domaines et langues
    # ------------------------------------------------------------------
    p(step_banner(2, 5, "Domaines Couverts & Langues"))
    domains = Counter(m.get("domain", "unknown") for m in metas)
    langs   = Counter(m.get("language", "unknown") for m in metas)
    p(table(
        ["Domaine", "Nb", "Langue", "Nb"],
        list(zip(
            [f"{d}  ({c})" for d, c in domains.most_common()],
            [""] * len(domains),
            [f"{l}  ({c})" for l, c in langs.most_common()],
            [""] * len(langs),
        ))[:max(len(domains), len(langs))],
    ))

    # ------------------------------------------------------------------
    # 3 — Trois exemples de documents complets
    # ------------------------------------------------------------------
    p(step_banner(3, 5, "Exemples de Documents Scientifiques"))

    samples = [docs[i] for i in [0, len(docs) // 2, len(docs) - 1]]
    for idx, doc in enumerate(samples, 1):
        m = doc["metadata"]
        preview = doc["content"].replace("\n", " ")[:220]
        p(f"\n  --- Document exemple #{idx} ---")
        p(box([
            f"source_file  : {m['source_file']}",
            f"file_type    : {m['file_type']}",
            f"domain       : {m['domain']}",
            f"compound     : {m['compound_name']}",
            f"cell_type    : {m['cell_type']}",
            f"language     : {m['language']}",
            f"chunk_index  : {m['chunk_index']}  |  page : {m['page_number']}",
            f"confidence   : {m['confidence_extraction']}",
            sep(),
            f"Contenu      : {preview}...",
        ]))

    # ------------------------------------------------------------------
    # 4 — Top composés et types cellulaires
    # ------------------------------------------------------------------
    p(step_banner(4, 5, "Top Composes Identifies & Types Cellulaires"))

    compounds = Counter(
        m["compound_name"] for m in metas if m["compound_name"] != "unknown"
    )
    cell_types = Counter(
        m["cell_type"] for m in metas if m["cell_type"] != "unknown"
    )

    c_rows = [[name, cnt] for name, cnt in compounds.most_common(8)]
    ct_rows = [[name, cnt] for name, cnt in cell_types.most_common(8)]

    p("\n  Composes les plus frequents :")
    if c_rows:
        p(table(["Compose", "Mentions"], c_rows))
    else:
        p("  (aucun compose identifie — metadata compound_name=unknown)")

    p("\n  Types cellulaires les plus frequents :")
    if ct_rows:
        p(table(["Type cellulaire", "Mentions"], ct_rows))
    else:
        p("  (aucun type cellulaire identifie)")

    # ------------------------------------------------------------------
    # 5 — Qualite des donnees
    # ------------------------------------------------------------------
    p(step_banner(5, 5, "Indicateurs de Qualite du Jeu de Donnees"))

    known_compound = sum(1 for m in metas if m["compound_name"] != "unknown")
    known_cell     = sum(1 for m in metas if m["cell_type"] != "unknown")
    high_conf      = sum(1 for m in metas if m.get("confidence_extraction", 0) >= 0.9)

    p("")
    p(metric_line("Docs avec compose identifie",  known_compound / len(docs)))
    p(metric_line("Docs avec type cellulaire",    known_cell     / len(docs)))
    p(metric_line("Docs haute confiance (>= 0.9)", high_conf    / len(docs)))

    p(f"\n{sep()}")
    p("CONCLUSION : Le jeu de donnees contient des publications scientifiques")
    p("biomedical / pharmaceutiques avec metadonnees structurees, pret pour")
    p("l'indexation vectorielle et la recherche semantique.")
    p(sep())

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_txt("01_dataset_overview", "\n".join(out_lines))


if __name__ == "__main__":
    run()
