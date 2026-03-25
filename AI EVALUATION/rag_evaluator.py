"""
rag_evaluator.py  —  RAG Pipeline Evaluator for EDI 814 Chatbot
----------------------------------------------------------------
Evaluates 4 dimensions:
  1. Groundedness     — is the answer supported by retrieved chunks?
  2. Relevance        — does the answer address the question?
  3. Correctness      — how close is the answer to the ground-truth answer?
  4. Retrieval Quality — did the right chunks get retrieved?

Input:  eval_dataset.json  (see format below)
Output: eval_results.json + eval_results.csv

Dataset format (eval_dataset.json):
[
  {
    "question": "What is the ESCO record retention requirement?",
    "ground_truth_answer": "ESCOs must retain customer records for 2 years per UBP...",
    "expected_keywords": ["2 years", "UBP", "customer authorization"],   ← optional
    "expected_source_keywords": ["NY ESCO Doc", "UBP"]                   ← optional, for retrieval eval
  },
  ...
]

Environment variables (same .env as chatbot_app.py):
  AZURE_SEARCH_ENDPOINT
  AZURE_SEARCH_ADMIN_KEY
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_EMBED_DEPLOY
  AZURE_OPENAI_CHAT_DEPLOY
  SEARCH_INDEX_NAME

Run:
  python rag_evaluator.py --dataset eval_dataset.json --output eval_results
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Azure Client Init
# ─────────────────────────────────────────────────────────────────────────────

def get_clients():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from openai import AzureOpenAI

    endpoint   = os.environ["AZURE_SEARCH_ENDPOINT"]
    key        = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    index_name = os.environ.get("SEARCH_INDEX_NAME", "edi-documents")
    aoai_ep    = os.environ["AZURE_OPENAI_ENDPOINT"]
    aoai_key   = os.environ["AZURE_OPENAI_API_KEY"]

    search_client = SearchClient(endpoint, index_name, AzureKeyCredential(key))
    openai_client = AzureOpenAI(
        azure_endpoint=aoai_ep,
        api_key=aoai_key,
        api_version="2024-02-01",
    )
    embed_deploy = os.environ.get("AZURE_OPENAI_EMBED_DEPLOY", "text-embedding-3-small")
    chat_deploy  = os.environ.get("AZURE_OPENAI_CHAT_DEPLOY", "gpt-4o-mini")
    return search_client, openai_client, embed_deploy, chat_deploy


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — Retrieval + Generation (mirrors chatbot_app.py)
# ─────────────────────────────────────────────────────────────────────────────

def embed_query(text: str, client, deploy: str) -> list[float]:
    resp = client.embeddings.create(input=[text], model=deploy)
    return resp.data[0].embedding


def hybrid_search(
    query: str,
    search_client,
    openai_client,
    embed_deploy: str,
    top_k: int = 15,
) -> list[dict]:
    from azure.search.documents.models import VectorizedQuery

    vector = embed_query(query, openai_client, embed_deploy)
    vq = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top_k * 2,
        fields="content_vector",
    )
    results = search_client.search(
        search_text=query,
        vector_queries=[vq],
        search_fields=["section_title", "topic", "subtopic", "content"],
        query_type="semantic",
        semantic_configuration_name="semantic-config",
        query_caption="extractive",
        top=top_k,
        select=[
            "chunk_id", "source", "source_pdf_url", "source_pdf_name",
            "chunk_index", "page_start", "page_end",
            "section_title", "section", "topic", "subtopic",
            "content", "content_type",
        ],
    )
    hits = []
    for r in results:
        d = dict(r)
        d["_score"] = r.get("@search.reranker_score") or r.get("@search.score", 0)
        hits.append(d)
    return hits


def build_context(hits: list[dict]) -> str:
    """Same source-tier ordering as chatbot_app.py — NY ESCO Doc first."""
    def source_priority(h):
        name = (h.get("source_pdf_name") or h.get("source") or "").lower()
        return 0 if ("esco doc" in name or "esco operating" in name) else 1

    sorted_hits = sorted(hits, key=source_priority)
    parts = []
    for i, h in enumerate(sorted_hits, 1):
        ps, pe  = h.get("page_start"), h.get("page_end")
        page_info = f"(Pages {ps}–{pe})" if ps and pe and ps != pe else (f"(Page {ps})" if ps else "")
        section = h.get("section", h.get("section_title", ""))
        ctype   = h.get("content_type", "text").upper()
        parts.append(
            f"[SOURCE {i}] [{ctype}] {section} {page_info}\n{h['content']}\n"
        )
    return "\n---\n".join(parts)


GENERATION_SYSTEM_PROMPT = """
You are a Retail Energy Regulatory and Market Rules Assistant.
Answer questions accurately using ONLY the provided source chunks.
Never fabricate information not present in the sources.
"""

def generate_answer(
    question: str,
    hits: list[dict],
    openai_client,
    chat_deploy: str,
) -> str:
    context = build_context(hits)
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Source chunks:\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]
    response = openai_client.chat.completions.create(
        model=chat_deploy,
        messages=messages,
        temperature=0.1,
        max_tokens=1200,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluators
# ─────────────────────────────────────────────────────────────────────────────

EVALUATOR_SYSTEM_PROMPT = """
You are a strict, expert evaluator for a RAG system built on New York retail
energy regulatory documents.

You will be given an evaluation task. Always respond ONLY with a valid JSON
object. No preamble, no markdown fences. Just the raw JSON.
"""


def _llm_evaluate(prompt: str, openai_client, chat_deploy: str) -> dict:
    """
    Call GPT with an evaluation prompt. Returns parsed JSON dict.
    Falls back to error dict on any failure.
    """
    try:
        response = openai_client.chat.completions.create(
            model=chat_deploy,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e), "score": 0.0}


# ── 1. Groundedness ───────────────────────────────────────────────────────────

def evaluate_groundedness(
    question: str,
    answer: str,
    context_chunks: list[dict],
    openai_client,
    chat_deploy: str,
) -> dict:
    """
    Measures: is every claim in the answer supported by the retrieved chunks?
    Score 0.0–1.0. Returns score + reasoning + unsupported_claims list.
    """
    context_text = build_context(context_chunks[:6])  # top 6 chunks is enough

    prompt = f"""
Evaluate the GROUNDEDNESS of the answer below.

Groundedness means: every factual claim in the answer must be directly supported
by the provided source chunks. Penalise if the answer introduces facts, figures,
or rules NOT present in the chunks.

Question: {question}

Source Chunks:
{context_text}

Generated Answer:
{answer}

Return a JSON object with:
{{
  "score": <float 0.0–1.0>,
  "reasoning": "<one paragraph explanation>",
  "unsupported_claims": ["<claim 1>", "<claim 2>"]  // empty list if fully grounded
}}

Scoring guide:
  1.0 = every claim is directly supported by the chunks
  0.7 = mostly supported, minor extrapolation
  0.4 = some claims unsupported or loosely inferred
  0.0 = answer contradicts or ignores the chunks entirely
"""
    result = _llm_evaluate(prompt, openai_client, chat_deploy)
    result["dimension"] = "groundedness"
    return result


# ── 2. Relevance ──────────────────────────────────────────────────────────────

def evaluate_relevance(
    question: str,
    answer: str,
    openai_client,
    chat_deploy: str,
) -> dict:
    """
    Measures: does the answer actually address what the question is asking?
    Score 0.0–1.0.
    """
    prompt = f"""
Evaluate the RELEVANCE of the answer to the question.

Relevance means: the answer directly addresses the question asked, stays on topic,
and does not meander into unrelated regulatory details.

Question: {question}

Generated Answer:
{answer}

Return a JSON object with:
{{
  "score": <float 0.0–1.0>,
  "reasoning": "<one paragraph explanation>",
  "missing_aspects": ["<aspect the answer missed>"]  // empty if fully relevant
}}

Scoring guide:
  1.0 = answer directly and completely addresses the question
  0.7 = mostly relevant, minor tangents or omissions
  0.4 = partially relevant, key aspects of the question ignored
  0.0 = answer is entirely off-topic
"""
    result = _llm_evaluate(prompt, openai_client, chat_deploy)
    result["dimension"] = "relevance"
    return result


# ── 3. Correctness ────────────────────────────────────────────────────────────

def evaluate_correctness(
    question: str,
    generated_answer: str,
    ground_truth_answer: str,
    openai_client,
    chat_deploy: str,
) -> dict:
    """
    Measures: how semantically close is the generated answer to the ground truth?
    Uses LLM as judge (not exact string match) so paraphrasing is handled.
    Score 0.0–1.0.
    """
    prompt = f"""
Evaluate the CORRECTNESS of the generated answer by comparing it against the
ground truth answer.

Correctness measures factual and semantic alignment — paraphrasing is fine,
but key facts, figures, rules, and conclusions must match.

Question: {question}

Ground Truth Answer:
{ground_truth_answer}

Generated Answer:
{generated_answer}

Return a JSON object with:
{{
  "score": <float 0.0–1.0>,
  "reasoning": "<one paragraph explanation>",
  "correct_facts": ["<fact that matches ground truth>"],
  "incorrect_or_missing_facts": ["<fact that is wrong or absent>"]
}}

Scoring guide:
  1.0 = all key facts match ground truth (paraphrasing allowed)
  0.7 = most key facts correct, minor omissions
  0.4 = some correct facts but significant gaps or errors
  0.0 = contradicts or entirely misses the ground truth
"""
    result = _llm_evaluate(prompt, openai_client, chat_deploy)
    result["dimension"] = "correctness"
    return result


# ── 4. Retrieval Quality ──────────────────────────────────────────────────────

def evaluate_retrieval_quality(
    question: str,
    hits: list[dict],
    expected_source_keywords: list[str],
    openai_client,
    chat_deploy: str,
) -> dict:
    """
    Measures retrieval quality on two sub-dimensions:
      a) Keyword coverage: do retrieved chunks contain expected source keywords?
         (deterministic — no LLM call needed)
      b) Chunk relevance: LLM rates how well the top chunks match the question.

    Returns a combined score + per-chunk relevance scores.
    """
    # ── a) Keyword coverage (deterministic) ──────────────────────────────
    retrieved_sources = [
        (h.get("source_pdf_name") or h.get("source") or "").lower()
        for h in hits
    ]
    retrieved_content = " ".join(h.get("content", "").lower() for h in hits)
    all_text = " ".join(retrieved_sources) + " " + retrieved_content

    keyword_hits = []
    keyword_misses = []
    for kw in (expected_source_keywords or []):
        if kw.lower() in all_text:
            keyword_hits.append(kw)
        else:
            keyword_misses.append(kw)

    keyword_coverage = (
        len(keyword_hits) / len(expected_source_keywords)
        if expected_source_keywords else None
    )

    # ── b) LLM chunk relevance scoring ───────────────────────────────────
    chunks_text = ""
    for i, h in enumerate(hits[:8]):  # evaluate top 8 chunks
        src  = h.get("source_pdf_name", "unknown")
        prev = h.get("content", "")[:300]
        chunks_text += f"\n[CHUNK {i}] Source: {src}\n{prev}\n"

    prompt = f"""
Evaluate the RETRIEVAL QUALITY for this question.

For each retrieved chunk, score its relevance to the question (0.0–1.0).
Then give an overall retrieval score.

Question: {question}

Retrieved Chunks:
{chunks_text}

Return a JSON object with:
{{
  "overall_score": <float 0.0–1.0>,
  "reasoning": "<one paragraph explanation>",
  "chunk_scores": [<score for chunk 0>, <score for chunk 1>, ...],
  "top_chunk_is_relevant": <true|false>,
  "irrelevant_chunks_count": <int>
}}

Scoring guide for overall_score:
  1.0 = top chunks are directly relevant, no noise
  0.7 = mostly relevant, 1–2 off-topic chunks
  0.4 = mixed — some relevant chunks buried among noise
  0.0 = retrieved chunks are entirely unrelated to the question
"""
    llm_result = _llm_evaluate(prompt, openai_client, chat_deploy)

    return {
        "dimension": "retrieval_quality",
        "overall_score": llm_result.get("overall_score", 0.0),
        "reasoning": llm_result.get("reasoning", ""),
        "chunk_scores": llm_result.get("chunk_scores", []),
        "top_chunk_is_relevant": llm_result.get("top_chunk_is_relevant"),
        "irrelevant_chunks_count": llm_result.get("irrelevant_chunks_count"),
        "keyword_coverage": keyword_coverage,
        "keyword_hits": keyword_hits,
        "keyword_misses": keyword_misses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composite Score
# ─────────────────────────────────────────────────────────────────────────────

DIMENSION_WEIGHTS = {
    "groundedness":      0.30,
    "relevance":         0.25,
    "correctness":       0.30,
    "retrieval_quality": 0.15,
}

def compute_composite_score(eval_results: dict) -> float:
    """
    Weighted average of the 4 dimension scores.
    Skips a dimension if its score is missing/errored.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for dim, weight in DIMENSION_WEIGHTS.items():
        score = eval_results.get(dim, {}).get("score") or \
                eval_results.get(dim, {}).get("overall_score")
        if score is not None:
            weighted_sum  += score * weight
            total_weight  += weight
    return round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Run Full Eval on One Row
# ─────────────────────────────────────────────────────────────────────────────

def run_eval_row(
    row: dict,
    search_client,
    openai_client,
    embed_deploy: str,
    chat_deploy: str,
    top_k: int = 15,
    delay: float = 1.0,
) -> dict:
    """
    End-to-end eval for a single dataset row.
    Returns a structured result dict.
    """
    question             = row["question"]
    ground_truth_answer  = row["ground_truth_answer"]
    expected_keywords    = row.get("expected_keywords", [])
    expected_src_kw      = row.get("expected_source_keywords", [])

    print(f"\n  → Retrieving chunks…")
    hits = hybrid_search(question, search_client, openai_client, embed_deploy, top_k)
    time.sleep(delay)

    print(f"  → Generating answer ({len(hits)} chunks)…")
    generated_answer = generate_answer(question, hits, openai_client, chat_deploy)
    time.sleep(delay)

    print(f"  → Evaluating groundedness…")
    g_result = evaluate_groundedness(question, generated_answer, hits, openai_client, chat_deploy)
    time.sleep(delay)

    print(f"  → Evaluating relevance…")
    r_result = evaluate_relevance(question, generated_answer, openai_client, chat_deploy)
    time.sleep(delay)

    print(f"  → Evaluating correctness…")
    c_result = evaluate_correctness(question, generated_answer, ground_truth_answer, openai_client, chat_deploy)
    time.sleep(delay)

    print(f"  → Evaluating retrieval quality…")
    rq_result = evaluate_retrieval_quality(question, hits, expected_src_kw, openai_client, chat_deploy)
    time.sleep(delay)

    # ── Expected keyword hit rate (simple string match on generated answer) ──
    answer_lower = generated_answer.lower()
    kw_hits   = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    kw_misses = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    kw_hit_rate = len(kw_hits) / len(expected_keywords) if expected_keywords else None

    eval_results = {
        "groundedness":      g_result,
        "relevance":         r_result,
        "correctness":       c_result,
        "retrieval_quality": rq_result,
    }

    composite = compute_composite_score(eval_results)

    return {
        "question":              question,
        "ground_truth_answer":   ground_truth_answer,
        "generated_answer":      generated_answer,
        "retrieved_sources":     [
            h.get("source_pdf_name", h.get("source", "")) for h in hits
        ],
        "num_chunks_retrieved":  len(hits),
        "eval": {
            **eval_results,
            "composite_score":   composite,
            "keyword_hit_rate":  kw_hit_rate,
            "keyword_hits":      kw_hits,
            "keyword_misses":    kw_misses,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate Summary
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(results: list[dict]) -> dict:
    def avg(vals):
        clean = [v for v in vals if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else None

    return {
        "total_rows":             len(results),
        "avg_groundedness":       avg([r["eval"]["groundedness"].get("score")       for r in results]),
        "avg_relevance":          avg([r["eval"]["relevance"].get("score")          for r in results]),
        "avg_correctness":        avg([r["eval"]["correctness"].get("score")        for r in results]),
        "avg_retrieval_quality":  avg([r["eval"]["retrieval_quality"].get("overall_score") for r in results]),
        "avg_composite_score":    avg([r["eval"]["composite_score"]                 for r in results]),
        "avg_keyword_hit_rate":   avg([r["eval"]["keyword_hit_rate"]                for r in results]),
        "avg_chunks_retrieved":   avg([r["num_chunks_retrieved"]                    for r in results]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save Outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_json(results: list[dict], summary: dict, output_path: str):
    out = {
        "metadata": {
            "run_timestamp": datetime.utcnow().isoformat() + "Z",
            "total_rows":    len(results),
            "weights":       DIMENSION_WEIGHTS,
        },
        "summary": summary,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ JSON saved → {output_path}")


def save_csv(results: list[dict], summary: dict, output_path: str):
    rows = []
    for r in results:
        ev = r["eval"]
        rows.append({
            "question":              r["question"],
            "ground_truth_answer":   r["ground_truth_answer"],
            "generated_answer":      r["generated_answer"],
            "retrieved_sources":     " | ".join(r["retrieved_sources"]),
            "num_chunks":            r["num_chunks_retrieved"],
            # Scores
            "groundedness_score":    ev["groundedness"].get("score"),
            "relevance_score":       ev["relevance"].get("score"),
            "correctness_score":     ev["correctness"].get("score"),
            "retrieval_quality_score": ev["retrieval_quality"].get("overall_score"),
            "composite_score":       ev["composite_score"],
            "keyword_hit_rate":      ev["keyword_hit_rate"],
            # Reasoning
            "groundedness_reasoning": ev["groundedness"].get("reasoning", ""),
            "relevance_reasoning":    ev["relevance"].get("reasoning", ""),
            "correctness_reasoning":  ev["correctness"].get("reasoning", ""),
            "retrieval_reasoning":    ev["retrieval_quality"].get("reasoning", ""),
            # Issues
            "unsupported_claims":    " | ".join(ev["groundedness"].get("unsupported_claims", [])),
            "missing_aspects":       " | ".join(ev["relevance"].get("missing_aspects", [])),
            "incorrect_facts":       " | ".join(ev["correctness"].get("incorrect_or_missing_facts", [])),
            "keyword_misses":        " | ".join(ev["keyword_misses"]),
            "retrieval_kw_misses":   " | ".join(ev["retrieval_quality"].get("keyword_misses", [])),
        })

    fieldnames = list(rows[0].keys()) if rows else []

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ CSV saved → {output_path}")

    # Print summary table to console
    print("\n" + "═" * 60)
    print("  EVALUATION SUMMARY")
    print("═" * 60)
    for k, v in summary.items():
        label = k.replace("_", " ").title()
        val   = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {label:<30} {val}")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluator")
    parser.add_argument(
        "--dataset", "-d",
        default="eval_dataset.json",
        help="Path to eval dataset JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        default="eval_results",
        help="Output file stem (without extension). Produces <stem>.json and <stem>.csv",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=15,
        help="Number of chunks to retrieve per question (default: 15)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls to avoid rate limits (default: 1.0)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Only evaluate the first N rows (useful for quick sanity checks)",
    )
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if args.limit:
        dataset = dataset[: args.limit]

    print(f"\n🔬 RAG Evaluator")
    print(f"   Dataset  : {dataset_path}  ({len(dataset)} rows)")
    print(f"   Output   : {args.output}.json / {args.output}.csv")
    print(f"   Top-K    : {args.top_k}")
    print(f"   Delay    : {args.delay}s\n")

    # ── Init clients ──────────────────────────────────────────────────────
    search_client, openai_client, embed_deploy, chat_deploy = get_clients()

    # ── Run evaluation ────────────────────────────────────────────────────
    all_results = []
    for idx, row in enumerate(dataset, 1):
        print(f"[{idx}/{len(dataset)}] {row['question'][:80]}…")
        try:
            result = run_eval_row(
                row=row,
                search_client=search_client,
                openai_client=openai_client,
                embed_deploy=embed_deploy,
                chat_deploy=chat_deploy,
                top_k=args.top_k,
                delay=args.delay,
            )
            all_results.append(result)
            print(f"  ✓ Composite score: {result['eval']['composite_score']:.4f}")
        except Exception as e:
            print(f"  ✗ Error on row {idx}: {e}")
            all_results.append({
                "question":            row.get("question", ""),
                "ground_truth_answer": row.get("ground_truth_answer", ""),
                "generated_answer":    "ERROR",
                "retrieved_sources":   [],
                "num_chunks_retrieved": 0,
                "eval":                {"error": str(e), "composite_score": 0.0},
            })

    # ── Save results ──────────────────────────────────────────────────────
    summary = compute_summary(all_results)
    save_json(all_results, summary, f"{args.output}.json")
    save_csv(all_results,  summary, f"{args.output}.csv")


if __name__ == "__main__":
    main()
