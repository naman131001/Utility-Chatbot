"""
rag_evaluator.py  —  RAG Evaluation for Utility Chatbot
---------------------------------------------------------
Runs azure-ai-evaluation metrics against your chatbot pipeline:
  GroundednessEvaluator  — hallucination detection
  RelevanceEvaluator     — did the answer address the query?
  RetrievalEvaluator     — did the search surface the right chunks?
  CoherenceEvaluator     — is the response logically structured?

Two modes:
  1. GENERATE mode  : runs your live pipeline on a question set → saves eval_dataset.jsonl
  2. EVALUATE mode  : loads eval_dataset.jsonl → runs all four evaluators → saves results

Usage:
  # Step 1 — generate eval dataset from live pipeline
  python rag_evaluator.py --mode generate --questions questions.txt

  # Step 2 — run evaluators on the generated dataset
  python rag_evaluator.py --mode evaluate

  # One-shot (generate + evaluate in sequence)
  python rag_evaluator.py --mode both --questions questions.txt

Environment variables (same as chatbot_app.py):
  AZURE_SEARCH_ENDPOINT
  AZURE_SEARCH_ADMIN_KEY
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_EMBED_DEPLOY   (default: text-embedding-3-small)
  AZURE_OPENAI_CHAT_DEPLOY    (default: gpt-4o-mini)
  SEARCH_INDEX_NAME           (default: edi-documents)

Output files:
  eval_dataset.jsonl    — {query, response, context, ground_truth?}
  eval_results.json     — per-row scores + aggregate summary
  eval_summary.txt      — human-readable summary table
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Default eval questions (edit or replace with questions.txt)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_QUESTIONS = [
    "When a REP sends an enrollment transaction to a utility, what data fields are required in the 814? "
]


# ─────────────────────────────────────────────────────────────────────────────
# Imports — mirrors chatbot_app.py
# ─────────────────────────────────────────────────────────────────────────────

def _get_clients():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from openai import AzureOpenAI

    endpoint   = os.environ["AZURE_SEARCH_ENDPOINT"]
    key        = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    index_name = os.environ.get("SEARCH_INDEX_NAME", "edi-documents")
    aoai_ep    = os.environ["AZURE_OPENAI_ENDPOINT"]
    aoai_key   = os.environ["AZURE_OPENAI_API_KEY"]

    cred          = AzureKeyCredential(key)
    search_client = SearchClient(endpoint, index_name, cred)
    openai_client = AzureOpenAI(
        azure_endpoint=aoai_ep,
        api_key=aoai_key,
        api_version="2024-02-01",
    )
    embed_deploy = os.environ.get("AZURE_OPENAI_EMBED_DEPLOY", "text-embedding-3-small")
    chat_deploy  = os.environ.get("AZURE_OPENAI_CHAT_DEPLOY", "gpt-4o-mini")
    return search_client, openai_client, embed_deploy, chat_deploy


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers (copied from chatbot_app.py — keep in sync)
# ─────────────────────────────────────────────────────────────────────────────

QUERY_REFINEMENT_PROMPT = """
You are a query classifier for a New York retail energy regulatory document system.
Given a user question, return ONLY a refined search query string that best captures
the user's intent for retrieval from a hybrid Azure AI Search index.
Output nothing else — no explanation, no JSON.
"""

SYSTEM_PROMPT = """
You are a Retail Energy Regulatory and Market Rules Assistant.
Your role is to provide accurate, market-specific, and utility-specific regulatory guidance
for retail energy operations in U.S. deregulated electricity and natural gas markets.

SOURCE HIERARCHY:
  TIER 1 — NY ESCO Doc / NY ESCO Operating Standards (always prefer for general UBP questions)
  TIER 2 — Utility-specific tariffs (use only when question names a utility)

Never generate regulatory information that is not supported by the provided sources.
"""


def embed_query(text: str, client, deploy: str) -> list[float]:
    resp = client.embeddings.create(input=[text], model=deploy)
    return resp.data[0].embedding


def hybrid_search(query: str, search_client, openai_client, embed_deploy: str, top_k: int = 15) -> list[dict]:
    from azure.search.documents.models import VectorizedQuery

    vector = embed_query(query, openai_client, embed_deploy)
    vq = VectorizedQuery(vector=vector, k_nearest_neighbors=top_k * 2, fields="content_vector")

    results = search_client.search(
        search_text=query,
        vector_queries=[vq],
        search_fields=["section_title", "topic", "subtopic", "content"],
        query_type="semantic",
        semantic_configuration_name="semantic-config",
        query_caption="extractive",
        query_answer="extractive",
        top=top_k,
        select=[
            "chunk_id", "source", "source_pdf_url", "source_pdf_name",
            "chunk_index", "page_start", "page_end",
            "section_title", "section", "topic", "subtopic",
            "content", "content_type", "is_table", "is_figure", "metadata_json",
        ],
    )

    hits = []
    for r in results:
        d = dict(r)
        d["_score"] = r.get("@search.reranker_score") or r.get("@search.score", 0)
        hits.append(d)
    return hits


def rerank_hits(query: str, hits: list[dict], openai_client, chat_deploy: str, top_n: int = 8) -> list[dict]:
    if not hits:
        return hits

    chunks_text = ""
    for i, h in enumerate(hits):
        content_preview = h.get("content", "")[:400]
        source = h.get("source_pdf_name", "unknown")
        chunks_text += f"\n[CHUNK {i}] Source: {source}\n{content_preview}\n"

    prompt = f"""Rate each chunk's relevance to the query (0.0–1.0).
Query: {query}
Chunks:{chunks_text}
Respond ONLY with a JSON array of floats, one per chunk. Example: [0.9, 0.2, 0.7]"""

    try:
        response = openai_client.chat.completions.create(
            model=chat_deploy,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = re.sub(r"```[a-z]*", "", response.choices[0].message.content.strip()).strip("`")
        scores = json.loads(raw)
        if isinstance(scores, list) and len(scores) == len(hits):
            for i, h in enumerate(hits):
                h["_rerank_score"] = float(scores[i])
            return sorted(hits, key=lambda x: x.get("_rerank_score", 0.0), reverse=True)[:top_n]
    except Exception:
        pass
    return hits[:top_n]


def build_context(hits: list[dict]) -> str:
    def source_priority(h):
        name = (h.get("source_pdf_name") or h.get("source") or "").lower()
        return 0 if ("esco doc" in name or "esco operating" in name) else 1

    sorted_hits = sorted(hits, key=source_priority)
    parts = []
    for i, h in enumerate(sorted_hits, 1):
        ps, pe = h.get("page_start"), h.get("page_end")
        page_info = f"  (Page {ps})" if ps and ps == pe else (f"  (Pages {ps}–{pe})" if ps else "")
        section = h.get("section", h.get("section_title", ""))
        ctype = h.get("content_type", "text").upper()
        parts.append(f"[SOURCE {i}] [{ctype}] {section}{page_info}\n{h['content']}\n")
    return "\n---\n".join(parts)


def generate_answer(question: str, hits: list[dict], openai_client, chat_deploy: str) -> str:
    context = build_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Use the following source chunks to answer the question.\n\n"
                f"=== SOURCE CHUNKS ===\n{context}\n\n"
                f"=== QUESTION ===\n{question}"
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
# GENERATE: run pipeline on questions → save eval_dataset.jsonl
# ─────────────────────────────────────────────────────────────────────────────

def generate_eval_dataset(
    questions: list[str],
    output_path: str = "eval_dataset.jsonl",
    top_k: int = 15,
    top_n_rerank: int = 8,
) -> list[dict]:
    print(f"\n{'='*60}")
    print("GENERATE MODE — running pipeline on questions")
    print(f"{'='*60}\n")

    search_client, openai_client, embed_deploy, chat_deploy = _get_clients()
    rows = []

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question[:70]}...")

        try:
            # Step 1: Hybrid search
            hits = hybrid_search(
                query=question,
                search_client=search_client,
                openai_client=openai_client,
                embed_deploy=embed_deploy,
                top_k=top_k,
            )

            # Step 2: Rerank
            hits = rerank_hits(
                query=question,
                hits=hits,
                openai_client=openai_client,
                chat_deploy=chat_deploy,
                top_n=top_n_rerank,
            )

            # Step 3: Build context string (what the LLM actually sees)
            context = build_context(hits)

            # Step 4: Generate answer
            response = generate_answer(
                question=question,
                hits=hits,
                openai_client=openai_client,
                chat_deploy=chat_deploy,
            )

            row = {
                "query": question,
                "response": response,
                "context": context,
                # ground_truth: fill in manually for NLP metrics (F1, ROUGE)
                # Leave blank if you don't have reference answers yet
                "ground_truth": "",
                # Metadata — not used by evaluators but useful for debugging
                "_num_chunks": len(hits),
                "_sources": [h.get("source_pdf_name", "") for h in hits],
            }
            rows.append(row)
            print(f"    ✓  {len(hits)} chunks → {len(response)} chars")

        except Exception as e:
            print(f"    ✗  Error: {e}")
            rows.append({
                "query": question,
                "response": f"ERROR: {e}",
                "context": "",
                "ground_truth": "",
                "_num_chunks": 0,
                "_sources": [],
            })

        # Avoid rate-limiting
        time.sleep(1)

    # Save
    Path(output_path).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    print(f"\n✅ Saved {len(rows)} rows → {output_path}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE: run all four evaluators on eval_dataset.jsonl
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluators(
    dataset_path: str = "eval_dataset.jsonl",
    results_path: str = "eval_results.json",
    summary_path: str = "eval_summary.txt",
):
    print(f"\n{'='*60}")
    print("EVALUATE MODE — running azure-ai-evaluation metrics")
    print(f"{'='*60}\n")

    try:
        from azure.ai.evaluation import (
            evaluate,
            GroundednessEvaluator,
            RelevanceEvaluator,
            RetrievalEvaluator,
            CoherenceEvaluator,
        )
    except ImportError:
        print("ERROR: azure-ai-evaluation not installed.")
        print("Run: pip install azure-ai-evaluation")
        sys.exit(1)

    # Judge model config — uses your same Azure OpenAI deployment
    model_config = {
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_key":        os.environ["AZURE_OPENAI_API_KEY"],
        "azure_deployment": os.environ.get("AZURE_OPENAI_CHAT_DEPLOY", "gpt-4o-mini"),
        "api_version": "2024-02-01",
    }

    print(f"Judge model : {model_config['azure_deployment']}")
    print(f"Dataset     : {dataset_path}\n")

    # ── Evaluator instances ────────────────────────────────────────────────
    groundedness = GroundednessEvaluator(model_config=model_config)
    relevance    = RelevanceEvaluator(model_config=model_config)
    retrieval    = RetrievalEvaluator(model_config=model_config)
    coherence    = CoherenceEvaluator(model_config=model_config)

    # ── Column mappings ────────────────────────────────────────────────────
    # Maps evaluator input names → JSONL column names
    evaluators = {
        "groundedness": groundedness,
        "relevance":    relevance,
        "retrieval":    retrieval,
        "coherence":    coherence,
    }

    evaluator_config = {
        "groundedness": {
            "column_mapping": {
                "query":    "${data.query}",
                "response": "${data.response}",
                "context":  "${data.context}",
            }
        },
        "relevance": {
            "column_mapping": {
                "query":    "${data.query}",
                "response": "${data.response}",
                "context":  "${data.context}",
            }
        },
        "retrieval": {
            "column_mapping": {
                "query":   "${data.query}",
                "context": "${data.context}",
            }
        },
        "coherence": {
            "column_mapping": {
                "query":    "${data.query}",
                "response": "${data.response}",
            }
        },
    }

    # ── Run evaluation ─────────────────────────────────────────────────────
    print("Running evaluators (this may take a few minutes)…\n")

    results = evaluate(
        data=dataset_path,
        evaluators=evaluators,
        evaluator_config=evaluator_config,
        # Optional: log to Azure AI Foundry portal
        # azure_ai_project={
        #     "subscription_id": os.environ.get("AZURE_SUBSCRIPTION_ID", ""),
        #     "resource_group_name": os.environ.get("AZURE_RESOURCE_GROUP", ""),
        #     "project_name": os.environ.get("AZURE_AI_PROJECT_NAME", ""),
        # },
        output_path=results_path,
    )

    # ── Print + save summary ───────────────────────────────────────────────
    _print_summary(results, summary_path)
    print(f"\n✅ Full results → {results_path}")
    print(f"✅ Summary      → {summary_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

SCORE_THRESHOLDS = {
    "groundedness": {"good": 4.0, "warn": 3.0},
    "relevance":    {"good": 4.0, "warn": 3.0},
    "retrieval":    {"good": 4.0, "warn": 3.0},
    "coherence":    {"good": 4.0, "warn": 3.0},
}


def _rating(metric: str, score: float) -> str:
    t = SCORE_THRESHOLDS.get(metric, {"good": 4.0, "warn": 3.0})
    if score >= t["good"]:  return "✅ GOOD"
    if score >= t["warn"]:  return "⚠️  FAIR"
    return "❌ POOR"


def _print_summary(results, summary_path: str):
    metrics = results.get("metrics", {})

    lines = [
        "=" * 60,
        "UTILITY CHATBOT — RAG EVALUATION SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        f"{'Metric':<20} {'Avg Score':>12}  {'Rating':<15}  {'What it measures'}",
        "-" * 75,
    ]

    descriptions = {
        "groundedness": "Hallucination: answer backed by retrieved chunks?",
        "relevance":    "Answer addresses the user's actual question?",
        "retrieval":    "Retrieved chunks relevant to the query?",
        "coherence":    "Response logically structured and readable?",
    }

    # azure-ai-evaluation prefixes metric names, e.g. "groundedness.groundedness"
    found_any = False
    for metric in ["groundedness", "relevance", "retrieval", "coherence"]:
        # Try common key patterns
        score = None
        for key in [metric, f"{metric}.{metric}", f"mean_{metric}"]:
            if key in metrics:
                score = metrics[key]
                break
        if score is None:
            # Fallback: search for any key containing the metric name
            for k, v in metrics.items():
                if metric in k.lower() and isinstance(v, (int, float)):
                    score = v
                    break

        if score is not None:
            found_any = True
            rating = _rating(metric, score)
            desc   = descriptions.get(metric, "")
            lines.append(f"  {metric:<18} {score:>8.2f} / 5   {rating:<15}  {desc}")
        else:
            lines.append(f"  {metric:<18} {'N/A':>10}   {'—':<15}  {descriptions.get(metric, '')}")

    lines += [
        "",
        "-" * 75,
        "SCORE GUIDE: 5 = excellent  |  4+ = good  |  3–4 = fair  |  <3 = needs work",
        "",
        "TROUBLESHOOTING:",
        "  Low groundedness → model is hallucinating beyond retrieved chunks",
        "    Fix: tighten system prompt, lower temperature, improve reranking",
        "  Low relevance    → answer misses the point of the question",
        "    Fix: review query refinement logic, improve chunking granularity",
        "  Low retrieval    → wrong chunks being surfaced from Azure AI Search",
        "    Fix: tune semantic config, BM25 field weights, embedding model",
        "  Low coherence    → response is disjointed or poorly formatted",
        "    Fix: adjust SYSTEM_PROMPT format guidelines, raise max_tokens",
        "=" * 60,
    ]

    summary_text = "\n".join(lines)
    print(summary_text)
    Path(summary_path).write_text(summary_text, encoding="utf-8")

    if not found_any:
        print("\nWARNING: Could not parse metric scores from results.")
        print("Raw metrics keys:", list(metrics.keys()) if metrics else "(empty)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def load_questions(questions_file: Optional[str]) -> list[str]:
    if questions_file and Path(questions_file).exists():
        lines = Path(questions_file).read_text(encoding="utf-8").splitlines()
        questions = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
        print(f"Loaded {len(questions)} questions from {questions_file}")
        return questions
    print(f"Using {len(DEFAULT_QUESTIONS)} default questions")
    return DEFAULT_QUESTIONS


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluator for Utility Chatbot")
    parser.add_argument(
        "--mode",
        choices=["generate", "evaluate", "both"],
        default="both",
        help="generate: run pipeline → JSONL | evaluate: run metrics | both: do both",
    )
    parser.add_argument(
        "--questions",
        default=None,
        help="Path to .txt file with one question per line (optional)",
    )
    parser.add_argument(
        "--dataset",
        default="eval_dataset.jsonl",
        help="Path to input/output JSONL dataset (default: eval_dataset.jsonl)",
    )
    parser.add_argument(
        "--results",
        default="eval_results.json",
        help="Path to output results JSON (default: eval_results.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of chunks to retrieve from Azure AI Search (default: 15)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of chunks to keep after reranking (default: 8)",
    )
    args = parser.parse_args()

    if args.mode in ("generate", "both"):
        questions = load_questions(args.questions)
        generate_eval_dataset(
            questions=questions,
            output_path=args.dataset,
            top_k=args.top_k,
            top_n_rerank=args.top_n,
        )

    if args.mode in ("evaluate", "both"):
        if not Path(args.dataset).exists():
            print(f"ERROR: Dataset not found: {args.dataset}")
            print("Run with --mode generate first.")
            sys.exit(1)
        run_evaluators(
            dataset_path=args.dataset,
            results_path=args.results,
            summary_path="eval_summary.txt",
        )


if __name__ == "__main__":
    main()