"""
Document parser using unstructured.io — supports PDF (with OCR), PPTX, DOCX,
TXT, HTML, XLSX.

Usage:
  1. Drop your files into data/raw/
  2. python ingestion/parser.py
  3. Then run ingestion/indexer.py to push chunks into Qdrant

Output: data/documents.json  (same schema as load_pubmedqa.py, compatible
        with ingestion/indexer.py)
"""

import json
import re
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "documents.json"

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt", ".html", ".xlsx"}

MAX_CHUNK_CHARS = 1500   # hard ceiling per chunk
NEW_AFTER_N_CHARS = 1200  # soft target — start new chunk after this many chars
COMBINE_UNDER = 200       # merge tiny fragments into the previous chunk

_EXT_TO_TYPE = {
    ".pdf": "pdf", ".pptx": "pptx", ".docx": "docx",
    ".txt": "txt", ".html": "html", ".xlsx": "xlsx",
}


# ---------------------------------------------------------------------------
# Metadata extraction helpers (heuristic, same logic as load_pubmedqa.py)
# ---------------------------------------------------------------------------

def _extract_compound(text: str) -> str:
    patterns = [
        r'\b([A-Z][a-z]+(?:inib|mab|zumab|ximab|tinib|ciclib|parib|rafenib|'
        r'lisib|metinib|lukast|sartan|pril|olol|statin))\b',
        r'\b(aspirin|ibuprofen|metformin|warfarin|heparin|insulin|penicillin|'
        r'amoxicillin|ciprofloxacin|doxorubicin|cisplatin|tamoxifen|paclitaxel|'
        r'docetaxel|gemcitabine|carboplatin|oxaliplatin|bevacizumab|'
        r'trastuzumab|rituximab)\b',
        r'(?:treatment with|treated with|administered|receiving)\s+([A-Z][a-z]{3,})',
        r'([A-Z][a-z]+)\s+\([A-Z]{2,5}\)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return "unknown"


def _extract_cell_type(text: str) -> str:
    patterns = [
        r'\b(HeLa|HEK[- ]?293|MCF-7|A549|Jurkat|CHO|PC-3|LNCaP|U87|T47D|MDA-MB-231)\b',
        r'\b([A-Za-z]+(?:\s+[A-Za-z]+)?\s+cell(?:s|\s+line))\b',
        r'\b([A-Za-z]+-[A-Za-z]+\s+cells?)\b',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return "unknown"


def _infer_year(path: Path, text: str) -> int:
    m = re.search(r'\b(19[89]\d|20[012]\d)\b', text[:500])
    if m:
        return int(m.group(1))
    m = re.search(r'\b(19[89]\d|20[012]\d)\b', path.stem)
    if m:
        return int(m.group(1))
    return 0


# ---------------------------------------------------------------------------
# Per-file parser
# ---------------------------------------------------------------------------

def parse_file(file_path: Path) -> list[dict]:
    """Return a list of chunk dicts ready for indexer.py."""
    elements = partition(
        filename=str(file_path),
        strategy="hi_res",        # uses OCR when needed (PDFs, scanned images)
        languages=["eng", "fra"], # Tesseract language packs
        include_page_breaks=False,
    )

    chunks = chunk_by_title(
        elements,
        max_characters=MAX_CHUNK_CHARS,
        new_after_n_chars=NEW_AFTER_N_CHARS,
        combine_text_under_n_chars=COMBINE_UNDER,
    )

    parent_id = str(uuid4())
    file_type = _EXT_TO_TYPE.get(file_path.suffix.lower(), "unknown")
    docs = []

    for idx, chunk in enumerate(chunks):
        content = str(chunk).strip()
        if not content:
            continue

        # Page number is stored in unstructured element metadata
        page_num = 1
        if hasattr(chunk, "metadata") and chunk.metadata:
            page_num = getattr(chunk.metadata, "page_number", None) or 1

        full_text = " ".join(d for d in [content] if d)
        docs.append({
            "content": content,
            "metadata": {
                "doc_id": str(uuid4()),
                "source_file": file_path.name,
                "file_type": file_type,
                "date_creation": "unknown",
                "year": _infer_year(file_path, content),
                "compound_name": _extract_compound(content),
                "test_type": "unknown",
                "cell_type": _extract_cell_type(content),
                "domain": "pharmaceutical",
                "chunk_index": idx,
                "parent_doc_id": parent_id,
                "page_number": page_num,
                "chunk_type": type(chunk).__name__.lower(),
                "language": "en",
                "confidence_extraction": 0.90,
            },
        })

    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True)
        print(f"Dossier cree : {RAW_DIR}")
        print("Deposez vos fichiers (PDF, PPTX, DOCX, TXT, HTML, XLSX) puis relancez.")
        return

    files = sorted(
        f for f in RAW_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"Aucun fichier supporte dans {RAW_DIR}")
        print(f"Formats acceptes : {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return

    print(f"{len(files)} fichier(s) trouve(s) dans {RAW_DIR.name}/")
    print("-" * 50)

    all_docs: list[dict] = []
    errors = []

    for file_path in files:
        print(f"  {file_path.name:<40}", end=" ", flush=True)
        try:
            docs = parse_file(file_path)
            all_docs.extend(docs)
            print(f"{len(docs):>4} chunks  OK")
        except Exception as exc:
            errors.append(file_path.name)
            print(f"ERREUR ({type(exc).__name__}: {exc})")

    print("-" * 50)
    print(f"Total : {len(all_docs)} chunks depuis {len(files) - len(errors)} fichier(s)")

    if not all_docs:
        print("Aucun chunk genere — documents.json non ecrase.")
        return

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print(f"Sauvegarde -> {OUTPUT_PATH.relative_to(ROOT)}")
    if errors:
        print(f"\nFichiers en erreur ({len(errors)}) : {', '.join(errors)}")
    print("\nEtape suivante : python ingestion/indexer.py")


if __name__ == "__main__":
    main()
