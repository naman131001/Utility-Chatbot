"""
retrieval.py — EDI 814 Chatbot Retrieval Pipeline (v2)
-------------------------------------------------------
Fixes:
  1. Query enrichment: extract utility/topic/state from question before searching
  2. Composite query string: prepend section metadata so BM25 matches content IN context
  3. Proper field boosting via scoringProfile or weighted search_fields
  4. Vector query embeds the ENRICHED query, not the raw question
  5. Remove broken cosine reranker — trust Azure Semantic Reranker exclusively
  6. Deduplicate by chunk_id BEFORE reranking so Azure sees a clean candidate set
  7. Single-pass search with higher top/knn instead of multi-query fan-out
"""

import json
import os
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Bump knn so the ANN pre-filter doesn't silently drop relevant chunks.
# Azure Semantic Reranker re-scores the top `SEMANTIC_TOP` of these.
KNN_CANDIDATES = 100   # vector ANN pre-filter pool
SEMANTIC_TOP   = 50    # how many go to semantic reranker
FINAL_TOP_K    = 5     # returned to the LLM

# Known utilities — used to detect & inject utility context into the query
KNOWN_UTILITIES = [
    "central hudson", "con edison", "coned",
    "national fuel", "national grid",
    "nyseg", "new york state electric",
    "orange and rockland", "o&r",
    "pseg", "pseg long island",
    "rochester gas", "rge",
]


# ─────────────────────────────────────────────────────────────────────────────
# Query enrichment
# ─────────────────────────────────────────────────────────────────────────────

def enrich_query(question: str, openai_client, chat_deploy: str) -> dict:
    """
    Extract structured context from the user question.
    Returns a dict: {utility, topic, state, enriched_query}

    The enriched_query is a single string that concatenates the extracted
    metadata with the question — this makes BM25 match on section-level
    fields more accurate, and improves the vector embedding alignment.
    """
    prompt = """You are a query parser for utility regulatory documents.

Given a user question, extract:
- utility: the utility company name if mentioned (else null)
- state: the US state if mentioned (else "New York" as default for this system)
- topic: the regulatory/operational topic (e.g. "nominations", "billing",
  "enrollment", "capacity release", "balancing", "EDI 814", "drop transaction")
- enriched_query: rewrite the question as a single declarative search phrase
  that includes utility name + topic + key terms, optimised for document search.
  Example: "Central Hudson gas nomination submission deadlines and procedures"

Return ONLY valid JSON. No markdown, no explanation.

Question: {question}
""".format(question=question)

    resp = openai_client.chat.completions.create(
        model=chat_deploy,
        messages=[
            {"role": "system", "content": "You extract structured search context from questions."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=200,
    )

    raw = resp.choices[0].message.content.strip()
    # Strip accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}

    # Fallback: if no enriched_query came back, build one heuristically
    if not parsed.get("enriched_query"):
        parsed["enriched_query"] = question

    # Inject utility if not caught by LLM but present in raw question
    if not parsed.get("utility"):
        q_lower = question.lower()
        for u in KNOWN_UTILITIES:
            if u in q_lower:
                parsed["utility"] = u.title()
                break

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed_query(text: str, client, deploy: str) -> list[float]:
    resp = client.embeddings.create(input=[text], model=deploy)
    return resp.data[0].embedding


# ─────────────────────────────────────────────────────────────────────────────
# Core search
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_search(
    enriched: dict,                  # output of enrich_query()
    search_client,
    openai_client,
    embed_deploy: str,
    top_k: int = FINAL_TOP_K,
    filter_content_type: Optional[str] = None,
    filter_source: Optional[str] = None,
) -> list[dict]:
    """
    Single-pass hybrid search using:
      - BM25 on content + section fields (weighted)
      - Vector ANN on the enriched query embedding
      - Azure Semantic Reranker as the final arbiter
    No downstream cosine reranking.
    """
    from azure.search.documents.models import VectorizedQuery

    enriched_query = enriched.get("enriched_query", "")
    utility        = enriched.get("utility", "")
    topic          = enriched.get("topic", "")

    # ── Build the BM25 search text ──────────────────────────────────────────
    # Prepend utility + topic so BM25 has section-level context baked in.
    # This dramatically reduces false positives from org-chart / contact chunks.
    bm25_parts = []
    if utility:
        bm25_parts.append(utility)
    if topic:
        bm25_parts.append(topic)
    bm25_parts.append(enriched_query)
    bm25_text = " ".join(bm25_parts)

    # ── Vector: embed the enriched query (not the raw question) ────────────
    vector = embed_query(enriched_query, openai_client, embed_deploy)
    vq = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=KNN_CANDIDATES,  # wider ANN pool = fewer missed chunks
        fields="content_vector",
    )

    # ── OData filter ────────────────────────────────────────────────────────
    filters = []
    if filter_content_type and filter_content_type != "All":
        filters.append(f"content_type eq '{filter_content_type.lower()}'")
    if filter_source and filter_source.strip():
        safe = filter_source.replace("'", "''")
        filters.append(f"source_pdf_name eq '{safe}'")
    # Auto-inject utility filter if detected and no manual source filter set
    if utility and not filter_source:
        safe_util = utility.replace("'", "''")
        filters.append(f"energy_utility_name eq '{safe_util}'")
    odata_filter = " and ".join(filters) if filters else None

    # ── Execute search ──────────────────────────────────────────────────────
    results = search_client.search(
        search_text=bm25_text,
        vector_queries=[vq],

        # Weighted fields: content is primary, section fields provide context signal.
        # Do NOT include topic/subtopic/section_title as equal BM25 fields —
        # they cause false positives when section titles happen to share keywords.
        search_fields=["content", "section_title"],

        query_type="semantic",
        semantic_configuration_name="semantic-config",
        query_caption="extractive",
        query_answer="extractive",

        # Give the semantic reranker a larger candidate pool to choose from.
        # It will re-score all SEMANTIC_TOP results and return them ranked.
        top=SEMANTIC_TOP,

        filter=odata_filter,
        select=[
            "chunk_id", "source", "source_pdf_url", "source_pdf_name",
            "chunk_index", "page_start", "page_end",
            "section_title", "section", "topic", "subtopic",
            "content", "content_type", "is_table", "is_figure",
            "energy_utility_name", "region", "metadata_json",
        ],
    )

    # ── Collect & deduplicate by chunk_id ───────────────────────────────────
    seen_ids = set()
    hits = []
    for r in results:
        cid = r.get("chunk_id", "")
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        d = dict(r)
        captions = r.get("@search.captions", [])
        if captions:
            d["_caption"] = captions[0].text

        # Prefer semantic reranker score; fall back to BM25 score
        d["_score"] = (
            r.get("@search.reranker_score")
            or r.get("@search.score", 0)
        )
        hits.append(d)

    # ── Sort by reranker score and return top_k ─────────────────────────────
    hits.sort(key=lambda x: x["_score"], reverse=True)
    return hits[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Optional: lightweight diversity filter
# ─────────────────────────────────────────────────────────────────────────────

def diversify(hits: list[dict], max_per_section: int = 2) -> list[dict]:
    """
    Prevent the same section from dominating the top-k results.
    Keeps up to `max_per_section` chunks per unique section path.
    Call this after hybrid_search if you notice answer repetition.
    """
    counts: dict[str, int] = {}
    out = []
    for h in hits:
        sec = h.get("section", h.get("section_title", ""))
        counts[sec] = counts.get(sec, 0) + 1
        if counts[sec] <= max_per_section:
            out.append(h)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point  (replaces the old rewrite_query + rerank_hits combo)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    question: str,
    search_client,
    openai_client,
    embed_deploy: str,
    chat_deploy: str,
    top_k: int = FINAL_TOP_K,
    filter_content_type: Optional[str] = None,
    filter_source: Optional[str] = None,
    apply_diversity: bool = True,
) -> tuple[list[dict], dict]:
    """
    Full retrieval pipeline. Returns (hits, enriched_meta).

    enriched_meta is the parsed query context dict — useful for displaying
    "Searching for: Central Hudson | nominations" in the UI.

    Usage in chatbot_app.py:
        hits, meta = retrieve(
            question=user_input,
            search_client=search_client,
            openai_client=openai_client,
            embed_deploy=embed_deploy,
            chat_deploy=chat_deploy,
            top_k=top_k,
            filter_content_type=filter_type if filter_type != "All" else None,
            filter_source=filter_doc if filter_doc.strip() else None,
        )
    """
    # Step 1: Understand the question
    enriched = enrich_query(question, openai_client, chat_deploy)

    # Step 2: Single-pass hybrid search with semantic reranking
    hits = hybrid_search(
        enriched=enriched,
        search_client=search_client,
        openai_client=openai_client,
        embed_deploy=embed_deploy,
        top_k=top_k * 2,          # fetch 2x then optionally diversify down to top_k
        filter_content_type=filter_content_type,
        filter_source=filter_source,
    )

    # Step 3: Optional section diversity
    if apply_diversity:
        hits = diversify(hits, max_per_section=2)

    return hits[:top_k], enriched