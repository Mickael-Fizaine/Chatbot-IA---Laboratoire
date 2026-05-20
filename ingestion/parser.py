"""
Document parser — supports PDF, PPTX, DOCX, TXT, HTML, XLSX, CSV.

Parsing strategy per format:
  .txt  → read directly with Python (no spaCy dependency)
  .docx → python-docx (no spaCy dependency)
  .pdf  → pypdf (no unstructured_inference dependency)
  .csv  → Python csv module
  .pptx / .html / .xlsx → unstructured.io (fallback to empty on error)

Chunking: paragraph-based with a configurable max_chars ceiling.

Usage:
  1. Drop files into data/raw/
  2. python ingestion/parser.py
  3. python ingestion/indexer.py  (push chunks into Qdrant)

Output: data/documents.json  (same schema as load_large_dataset.py)
"""

import csv as csv_module
import json
import re
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "documents.json"

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt", ".html", ".xlsx", ".csv"}

MAX_CHUNK_CHARS = 1500
NEW_AFTER_N_CHARS = 1200
COMBINE_UNDER = 200

_EXT_TO_TYPE = {
    ".pdf": "pdf", ".pptx": "pptx", ".docx": "docx",
    ".txt": "txt", ".html": "html", ".xlsx": "xlsx", ".csv": "csv",
}


# ---------------------------------------------------------------------------
# Metadata extraction helpers
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
# Text chunking (used for formats parsed without unstructured)
# ---------------------------------------------------------------------------

def _chunk_paragraphs(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text at double-newlines; merge short paragraphs; hard-cut long ones."""
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0
        # Hard-cut paragraphs that exceed max_chars on their own
        while len(para) > max_chars:
            chunks.append(para[:max_chars])
            para = para[max_chars:]
        current_parts.append(para)
        current_len += len(para)

    if current_parts:
        chunks.append("\n\n".join(current_parts))
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Format-specific extractors → list of (text_chunk, page_number)
# ---------------------------------------------------------------------------

def _extract_txt(path: Path) -> list[tuple[str, int]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [(chunk, 1) for chunk in _chunk_paragraphs(text)]


def _extract_docx(path: Path) -> list[tuple[str, int]]:
    from docx import Document
    doc = Document(str(path))
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(chunk, 1) for chunk in _chunk_paragraphs(text)]


def _extract_pdf(path: Path) -> list[tuple[str, int]]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    result: list[tuple[str, int]] = []
    for page_num, page in enumerate(reader.pages, 1):
        raw = page.extract_text() or ""
        if raw.strip():
            for chunk in _chunk_paragraphs(raw):
                result.append((chunk, page_num))
    return result


def _extract_csv(path: Path) -> list[tuple[str, int]]:
    rows: list[str] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv_module.reader(f)
        for row in reader:
            line = ", ".join(str(c) for c in row if str(c).strip())
            if line:
                rows.append(line)
    text = "\n".join(rows)
    return [(chunk, 1) for chunk in _chunk_paragraphs(text)]


def _extract_with_unstructured(path: Path) -> list[tuple[str, int]]:
    """Fallback for PPTX / HTML / XLSX using unstructured.io."""
    from unstructured.partition.auto import partition
    from unstructured.chunking.title import chunk_by_title

    elements = partition(filename=str(path), include_page_breaks=False)
    chunks = chunk_by_title(
        elements,
        max_characters=MAX_CHUNK_CHARS,
        new_after_n_chars=NEW_AFTER_N_CHARS,
        combine_text_under_n_chars=COMBINE_UNDER,
    )
    result: list[tuple[str, int]] = []
    for chunk in chunks:
        content = str(chunk).strip()
        if not content:
            continue
        page_num = 1
        if hasattr(chunk, "metadata") and chunk.metadata:
            page_num = getattr(chunk.metadata, "page_number", None) or 1
        result.append((content, page_num))
    return result


# ---------------------------------------------------------------------------
# Main per-file entry point
# ---------------------------------------------------------------------------

def parse_file(file_path: Path) -> list[dict]:
    """Return a list of chunk dicts ready for indexer.py."""
    ext = file_path.suffix.lower()

    if ext == ".txt":
        pairs = _extract_txt(file_path)
    elif ext == ".docx":
        pairs = _extract_docx(file_path)
    elif ext == ".pdf":
        pairs = _extract_pdf(file_path)
    elif ext == ".csv":
        pairs = _extract_csv(file_path)
    else:
        pairs = _extract_with_unstructured(file_path)

    parent_id = str(uuid4())
    file_type = _EXT_TO_TYPE.get(ext, "unknown")
    docs: list[dict] = []

    for idx, (content, page_num) in enumerate(pairs):
        if not content:
            continue
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
                "chunk_type": "text",
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
        print("Deposez vos fichiers (PDF, PPTX, DOCX, TXT, HTML, XLSX, CSV) puis relancez.")
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
        print(f"  {file_path.name:<42}", end=" ", flush=True)
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
