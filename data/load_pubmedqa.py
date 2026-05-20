# Version reduite : 300 documents PubMedQA pqa_labeled.
# Pour le dataset de production (20 000 docs, multi-formats) :
#   python data/load_large_dataset.py

import json
import re
from pathlib import Path
from uuid import uuid4

from datasets import load_dataset


def extract_compound_name(text: str) -> str:
    """Heuristic extraction of drug/compound names from biomedical text."""
    patterns = [
        # Drug suffix patterns (IUPAC-style endings common in pharma)
        r'\b([A-Z][a-z]+(?:inib|mab|zumab|ximab|tinib|ciclib|parib|rafenib|lisib|metinib|lukast|sartan|pril|olol|statin))\b',
        # Common small molecules
        r'\b(aspirin|ibuprofen|metformin|warfarin|heparin|insulin|penicillin|amoxicillin|'
        r'ciprofloxacin|doxorubicin|cisplatin|tamoxifen|paclitaxel|docetaxel|gemcitabine|'
        r'carboplatin|oxaliplatin|bevacizumab|trastuzumab|rituximab)\b',
        # "treatment with X" or "X treatment"
        r'(?:treatment with|treated with|administered|receiving)\s+([A-Z][a-z]{3,})',
        # Uppercase abbreviation with parentheses: "Imatinib (IM)"
        r'([A-Z][a-z]+)\s+\([A-Z]{2,5}\)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "unknown"


def extract_cell_type(text: str) -> str:
    """Heuristic extraction of cell type from biomedical text."""
    patterns = [
        # Known cell line names
        r'\b(HeLa|HEK[- ]?293|MCF-7|A549|Jurkat|CHO|PC-3|LNCaP|U87|T47D|MDA-MB-231)\b',
        # "X cells" or "X cell line"
        r'\b([A-Za-z]+(?:\s+[A-Za-z]+)?\s+cell(?:s|\s+line))\b',
        # "X-type cells"
        r'\b([A-Za-z]+-[A-Za-z]+\s+cells?)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "unknown"


def build_content(example: dict) -> str:
    ctx = example.get("context", {})
    if isinstance(ctx, dict):
        ctx_parts = ctx.get("contexts", [])
        context_str = " ".join(ctx_parts) if isinstance(ctx_parts, list) else str(ctx_parts)
    else:
        context_str = str(ctx)

    question = example.get("question", "")
    long_answer = example.get("long_answer", "")

    parts = [p for p in [context_str, f"Question: {question}", f"Answer: {long_answer}"] if p.strip()]
    return "\n\n".join(parts)


def build_document(example: dict) -> dict:
    pubmed_id = str(example.get("pubid", "unknown"))
    content = build_content(example)

    doc_id = str(uuid4())

    metadata = {
        "doc_id": doc_id,
        "source_file": f"pubmed_{pubmed_id}.pdf",
        "file_type": "pdf",
        "date_creation": "2020-01-01",
        "year": 2020,
        "compound_name": extract_compound_name(content),
        "test_type": "efficacy",
        "cell_type": extract_cell_type(content),
        "domain": "pharmaceutical",
        "chunk_index": 0,
        "parent_doc_id": doc_id,
        "page_number": 1,
        "chunk_type": "abstract",
        "language": "en",
        "confidence_extraction": 0.95,
    }

    return {"content": content, "metadata": metadata}


def main():
    print("Loading PubMedQA dataset (pqa_labeled)...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

    split = dataset["train"]
    examples = list(split.select(range(min(300, len(split)))))
    print(f"  {len(examples)} examples selected.")

    print("Building documents...")
    documents = [build_document(ex) for ex in examples]

    output_path = Path(__file__).parent / "documents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"\n--- Summary ---")
    print(f"Documents saved : {len(documents)}")
    print(f"Output file     : {output_path}")
    print(f"\nFirst document:")
    first = documents[0]
    preview = first["content"][:300].replace("\n", " ")
    print(f"  content (preview) : {preview}...")
    print(f"  metadata          : {json.dumps(first['metadata'], indent=4)}")


if __name__ == "__main__":
    main()
