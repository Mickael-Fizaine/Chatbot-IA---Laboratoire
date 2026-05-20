"""
DEMO 02 — Visualisation du Pipeline : Parsing → Chunking → Embedding → Qdrant
===============================================================================
Montre pas a pas comment un document passe de son format brut (PDF / PPTX / DOCX)
jusqu'a son stockage vectoriel dans Qdrant, pret pour la recherche semantique.

Prerequis : data/documents.json  +  Qdrant en cours d'execution (Docker)
Dataset    : python data/load_large_dataset.py  (20 000 docs, multi-formats)
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from demo.utils import (
    W, bar, box, header, save_txt, section, sep, step_banner, table
)

DOCS_PATH = ROOT / "data" / "documents.json"


def load_sample_doc(docs: list[dict]) -> dict:
    # prefer a doc with a known compound
    for d in docs:
        if d["metadata"]["compound_name"] != "unknown":
            return d
    return docs[0]


def show_parsing(doc: dict, out):
    out("\n  Simulation de la sortie unstructured.io sur un fichier PDF :")
    out("  (strategy='hi_res' → OCR active pour les PDFs scannes)\n")

    content = doc["content"]
    source  = doc["metadata"]["source_file"]

    # Split content into "elements" to simulate unstructured output
    parts = [p.strip() for p in content.split("\n\n") if p.strip()][:4]
    element_types = ["Title", "NarrativeText", "NarrativeText", "Table"]

    out(box([
        f"Fichier source : {source}",
        f"Strategie      : hi_res (OCR Tesseract pour scans)",
        f"Langues        : eng, fra",
        sep("─"),
        *[
            f"[{element_types[i]:<14}]  {parts[i][:55]}..."
            if len(parts[i]) > 55 else f"[{element_types[i]:<14}]  {parts[i]}"
            for i in range(min(len(parts), len(element_types)))
        ],
    ]))


def show_chunking(doc: dict, docs_sample: list[dict], out):
    # Simulate several chunks from the same parent document
    parent_id = doc["metadata"]["parent_doc_id"]
    related = [d for d in docs_sample if d["metadata"]["parent_doc_id"] == parent_id]
    if not related:
        related = [doc]

    total_chars = sum(len(d["content"]) for d in related)
    out(f"\n  Document brut   : 1 fichier  |  ~{total_chars:,} caracteres")
    out(f"  Apres chunking  : {len(related)} chunk(s)  |  ~{total_chars // max(len(related),1):,} car./chunk en moyenne")
    out(f"  Parametres      : max_characters=1500, new_after_n_chars=1200\n")

    rows = [
        [
            f"#{d['metadata']['chunk_index']}",
            d["metadata"]["chunk_type"],
            d["metadata"]["source_file"],
            str(d["metadata"]["page_number"]),
            f"{len(d['content']):,}",
            d["content"][:45] + "...",
        ]
        for d in related[:6]
    ]
    out(table(
        ["Chunk", "Type", "Source", "Page", "Chars", "Apercu"],
        rows,
        max_col=48,
    ))


def show_embedding(docs: list[dict], out):
    out("\n  Modele : gemini-embedding-001   Dimension : 3072")
    out("  Chaque chunk est transforme en un vecteur de 3072 nombres flottants.\n")

    # Try to load a real vector from Qdrant
    try:
        from qdrant_client import QdrantClient
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        client = QdrantClient(host=host, port=port)
        collection = os.getenv("COLLECTION_NAME", "lab_documents")
        results = client.query_points(
            collection_name=collection,
            query=[0.0] * 3072,
            limit=1,
            with_vectors=True,
        )
        if results.points:
            vec = results.points[0].vector
            snippet = [f"{v:.4f}" for v in vec[:8]]
            out(box([
                "Vecteur REEL extrait depuis Qdrant (8 premieres dimensions / 3072) :",
                "",
                f"  [ {', '.join(snippet)}, ... ]",
                "",
                f"  Norme L2 approx. : {sum(x**2 for x in vec)**0.5:.4f}",
                f"  Similarite utilisee : COSINE (normalisation automatique)",
            ]))
            return
    except Exception:
        pass

    # Fallback: illustrative values
    out(box([
        "Representation illustrative du vecteur (8 dim. / 3072) :",
        "",
        "  [ 0.0231, -0.0145, 0.0891, -0.0332, 0.0456, 0.0017, -0.0723, 0.0389, ... ]",
        "",
        "  Chaque dimension encode un aspect semantique du texte.",
        "  Deux chunks proches semantiquement ont des vecteurs similaires.",
        "  Similarite mesuree par COSINE dans Qdrant.",
        "",
        "  (Qdrant non accessible — valeurs illustratives)",
    ]))


def show_qdrant_point(doc: dict, out):
    m = doc["metadata"]
    out("\n  Structure d'un 'Point' Qdrant (1 chunk = 1 point) :\n")
    out(box([
        "{",
        f'  "id"      : 42,',
        f'  "vector"  : [0.0231, -0.0145, ...],   // 3072 floats',
        f'  "payload" : {{',
        f'    "content"      : "{doc["content"][:60]}...",',
        f'    "source_file"  : "{m["source_file"]}",',
        f'    "file_type"    : "{m["file_type"]}",',
        f'    "compound_name": "{m["compound_name"]}",',
        f'    "cell_type"    : "{m["cell_type"]}",',
        f'    "page_number"  : {m["page_number"]},',
        f'    "chunk_index"  : {m["chunk_index"]}',
        f'  }}',
        "}",
    ]))


def show_qdrant_stats(out):
    out("\n  Stats de la collection Qdrant 'lab_documents' :")
    try:
        from qdrant_client import QdrantClient
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        client = QdrantClient(host=host, port=port)
        collection = os.getenv("COLLECTION_NAME", "lab_documents")
        info = client.get_collection(collection)
        count = client.count(collection_name=collection).count
        cfg   = info.config.params.vectors
        out(box([
            f"  Nom de la collection : {collection}",
            f"  Nombre de points     : {count:,}  (= chunks indexes)",
            f"  Dimension vecteur    : {cfg.size}",
            f"  Distance             : {cfg.distance.value}",
            f"  Status               : {info.status.value}",
        ]))
    except Exception as exc:
        out(box([
            "Qdrant non accessible.",
            f"({type(exc).__name__}) — lancez Docker puis relancez ce script.",
        ]))


def run():
    out_lines: list[str] = []

    def p(text: str = ""):
        print(text)
        out_lines.append(text)

    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    doc = load_sample_doc(docs)

    p(header(
        "DEMO 02 — Pipeline : Parsing → Chunking → Embedding → Qdrant",
        "De la donnee brute au vecteur indexe et interrogeable"
    ))

    p(step_banner(1, 4, "Parsing du Fichier Brut (unstructured.io, OCR)"))
    show_parsing(doc, p)

    p(step_banner(2, 4, "Decoupage en Chunks Semantiques (chunk_by_title)"))
    show_chunking(doc, docs, p)

    p(step_banner(3, 4, "Vectorisation — Embedding gemini-embedding-001"))
    show_embedding(docs, p)

    p(step_banner(4, 4, "Stockage dans Qdrant"))
    show_qdrant_point(doc, p)
    show_qdrant_stats(p)

    p(f"\n{sep()}")
    p("CONCLUSION : Chaque fichier est parse, decoupe en ~4-10 chunks,")
    p("vectorise en 3072 dimensions et stocke dans Qdrant avec ses metadonnees.")
    p("La recherche peut ensuite filtrer par compound, cell_type, source_file, etc.")
    p(sep())

    save_txt("02_pipeline_viz", "\n".join(out_lines))


if __name__ == "__main__":
    run()
