"""
RAGAS evaluation pipeline for the Lab Chatbot.

Steps:
  1. Generate 20 question/ground_truth pairs from indexed documents via Gemini.
  2. Run process_query() on each question.
  3. Evaluate with RAGAS metrics (faithfulness, answer_relevancy,
     context_precision, context_recall).
  4. Print report and save evaluation/ragas_report.json.
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TESTSET_PATH = Path(__file__).parent / "testset.json"
REPORT_PATH = Path(__file__).parent / "ragas_report.json"
DOCS_PATH = ROOT / "data" / "documents.json"
N_SAMPLES = 20
INTER_CALL_SLEEP = 8   # seconds between Gemini generation calls (rate limit buffer)

_GENERATION_PROMPT_HEADER = (
    "Generate a precise question in English whose answer can be found in the text below. "
    "Return ONLY a valid JSON object without markdown, using EXACTLY these two keys:\n"
    '{"question": "the question here", "ground_truth": "the answer here"}\n\n'
    "Text:\n"
)


def _build_prompt(text: str) -> str:
    # Avoid str.format() — document text may contain braces like {IC50} or {CaCl2}
    # which Python's formatter would misinterpret as placeholder names.
    return _GENERATION_PROMPT_HEADER + text

# Key aliases Gemini sometimes uses instead of the canonical names
_QUESTION_KEYS = ("question", "query", "question_text", "q")
_GROUND_TRUTH_KEYS = ("ground_truth", "answer", "expected_answer", "correct_answer", "response", "ground truth")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    parsed = json.loads(text)
    # Normalize to canonical keys in case Gemini used an alias
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    question = next((parsed[k] for k in _QUESTION_KEYS if k in parsed), None)
    ground_truth = next((parsed[k] for k in _GROUND_TRUTH_KEYS if k in parsed), None)
    if question is None:
        raise KeyError(f"No question key in response keys={list(parsed.keys())}")
    if ground_truth is None:
        raise KeyError(f"No ground_truth key in response keys={list(parsed.keys())}")
    return {"question": str(question), "ground_truth": str(ground_truth)}


def _invoke_llm_with_retry(llm, messages: list, max_retries: int = 4) -> object:
    """Call a LangChain LLM with exponential backoff on 429 rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            if attempt == max_retries - 1 or not is_rate_limit:
                raise
            wait = 30 * (attempt + 1)
            print(f"  [rate-limit] retry {attempt + 1}/{max_retries}, waiting {wait}s...")
            time.sleep(wait)

# ---------------------------------------------------------------------------
# Step 1 — Generate testset
# ---------------------------------------------------------------------------

def generate_testset(llm) -> list[dict]:
    """Return 20 {question, ground_truth, source_file} dicts, loading from cache if available."""
    if TESTSET_PATH.exists():
        print(f"Loading existing testset from {TESTSET_PATH.name}")
        with open(TESTSET_PATH, encoding="utf-8") as f:
            return json.load(f)

    print(f"Generating testset from {DOCS_PATH.name} ...")
    with open(DOCS_PATH, encoding="utf-8") as f:
        all_docs = json.load(f)

    # Keep only docs with enough real content (skip metadata-only files)
    rich_docs = [d for d in all_docs if len(d.get("content", "")) >= 300]
    sample_docs = random.sample(rich_docs, min(N_SAMPLES, len(rich_docs)))
    testset = []

    from langchain_core.messages import HumanMessage

    for i, doc in enumerate(sample_docs, 1):
        content = doc.get("content", "")[:1200]
        source = doc.get("metadata", {}).get("source_file", f"doc_{i}")
        print(f"  [{i:02d}/{N_SAMPLES}] {source} ...", end=" ", flush=True)
        try:
            prompt = _build_prompt(content)
            response = _invoke_llm_with_retry(llm, [HumanMessage(content=prompt)])
            parsed = _parse_json_response(response.content)
            testset.append({
                "question": parsed["question"],
                "ground_truth": parsed["ground_truth"],
                "source_file": source,
            })
            print("OK")
        except Exception as exc:
            print(f"SKIP ({exc.__class__.__name__}: {exc})")
        time.sleep(INTER_CALL_SLEEP)

    with open(TESTSET_PATH, "w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)
    print(f"Testset saved: {len(testset)} pairs -> {TESTSET_PATH.name}\n")
    return testset

# ---------------------------------------------------------------------------
# Step 2 — Run pipeline on each question
# ---------------------------------------------------------------------------

def run_pipeline(testset: list[dict]) -> list[dict]:
    """Run process_query() on each question, return enriched records."""
    from graph.graph import process_query

    results = []
    print(f"Running pipeline on {len(testset)} questions ...")
    for i, item in enumerate(testset, 1):
        q = item["question"]
        print(f"  [{i:02d}/{len(testset)}] {q[:70]}...", end=" ", flush=True)
        try:
            r = process_query(q)
            results.append({
                "question": q,
                "ground_truth": item["ground_truth"],
                "source_file": item.get("source_file", ""),
                "answer": r.get("answer", ""),
                "contexts": [
                    doc.get("content", "") for doc in r.get("documents", [])
                ] or [item.get("ground_truth", "")],
                "query_type": r.get("query_type", ""),
                "relevance_score": r.get("relevance_score", 0.0),
                "hallucination_score": r.get("hallucination_score", 0.0),
            })
            print("OK")
        except Exception as exc:
            print(f"ERROR ({exc})")
            results.append({
                "question": q,
                "ground_truth": item["ground_truth"],
                "source_file": item.get("source_file", ""),
                "answer": "",
                "contexts": [item.get("ground_truth", "")],
                "query_type": "",
                "relevance_score": 0.0,
                "hallucination_score": 0.0,
            })
        time.sleep(INTER_CALL_SLEEP)
    return results

# ---------------------------------------------------------------------------
# Step 3 — RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas(pipeline_results: list[dict], llm, embeddings) -> object:
    """Build EvaluationDataset and run RAGAS metrics (RAGAS 0.4.x API)."""
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics.collections.faithfulness import Faithfulness
    from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
    from ragas.metrics.collections.context_precision import ContextPrecisionWithReference
    from ragas.metrics.collections.context_recall import ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    samples = []
    for r in pipeline_results:
        contexts = r["contexts"] if r["contexts"] else [r["ground_truth"]]
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"] or "Information non disponible dans la base de connaissances.",
                retrieved_contexts=contexts,
                reference=r["ground_truth"],
            )
        )

    dataset = EvaluationDataset(samples=samples)

    # RAGAS 0.4.x: metrics must be instantiated with llm (required positional arg)
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecisionWithReference(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    print("\nRunning RAGAS evaluation ...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        show_progress=True,
        raise_exceptions=False,
    )
    return result

# ---------------------------------------------------------------------------
# Step 4 — Display report
# ---------------------------------------------------------------------------

def display_report(ragas_result, pipeline_results: list[dict]) -> dict:
    # RAGAS 0.4.x: scores are in to_pandas(), no .scores.mean() attribute
    df = ragas_result.to_pandas()
    numeric_cols = df.select_dtypes(include="number").columns

    def _mean(col: str) -> float:
        return float(df[col].dropna().mean()) if col in numeric_cols else 0.0

    # Map RAGAS 0.4.x column names (may differ from 0.1.x)
    faithfulness_score = _mean("faithfulness")
    answer_relevancy_score = _mean("answer_relevancy")
    context_precision_score = _mean("context_precision_with_reference") or _mean("context_precision")
    context_recall_score = _mean("context_recall")
    avg = (faithfulness_score + answer_relevancy_score + context_precision_score + context_recall_score) / 4

    problematic = []
    if "faithfulness" in df.columns:
        for _, row in df.iterrows():
            val = row.get("faithfulness")
            if val is not None and float(val) < 0.5:
                problematic.append({
                    "question": row.get("user_input", ""),
                    "faithfulness": float(val),
                })

    sep = "=" * 50
    thin = "-" * 50
    print(f"\n{sep}")
    print("RAPPORT RAGAS -- Lab Chatbot")
    print(sep)
    print(f"Faithfulness      : {faithfulness_score:.2f}")
    print(f"Answer Relevancy  : {answer_relevancy_score:.2f}")
    print(f"Context Precision : {context_precision_score:.2f}")
    print(f"Context Recall    : {context_recall_score:.2f}")
    print(thin)
    print(f"Score moyen       : {avg:.2f}")

    if problematic:
        print("\nQuestions problematiques (faithfulness < 0.5) :")
        for p in problematic:
            print(f"  - \"{p['question'][:80]}\" -> score {p['faithfulness']:.2f}")
    else:
        print("\nAucune question avec faithfulness < 0.5")

    verdict = "Pipeline pret (score > 0.70)" if avg >= 0.70 else "Ajustements necessaires"
    print(f"\n{sep}")
    print(f"VERDICT : {verdict}")
    print(sep)

    report = {
        "metrics": {
            "faithfulness": faithfulness_score,
            "answer_relevancy": answer_relevancy_score,
            "context_precision": context_precision_score,
            "context_recall": context_recall_score,
            "average": avg,
            "ragas_columns": list(df.columns),
        },
        "verdict": verdict,
        "problematic_questions": problematic,
        "per_question": df.to_dict(orient="records") if not df.empty else [],
    }
    return report

# ---------------------------------------------------------------------------
# Step 5 — Save report
# ---------------------------------------------------------------------------

def save_report(report: dict) -> None:
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved -> {REPORT_PATH.name}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    testset = generate_testset(llm)
    if not testset:
        print("No testset generated, aborting.")
        sys.exit(1)

    pipeline_results = run_pipeline(testset)

    try:
        ragas_result = run_ragas(pipeline_results, llm, embeddings)
    except Exception as exc:
        import traceback
        print(f"\n[RAGAS ERROR] {exc}")
        traceback.print_exc()
        sys.exit(1)

    try:
        report = display_report(ragas_result, pipeline_results)
        save_report(report)
    except Exception as exc:
        import traceback
        print(f"\n[REPORT ERROR] {exc}")
        traceback.print_exc()
        # Still save raw pandas output for debugging
        try:
            df = ragas_result.to_pandas()
            print("\nRaw RAGAS scores:")
            print(df.to_string())
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
