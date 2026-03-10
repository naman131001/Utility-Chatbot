"""
retriever.py
------------
Hybrid retriever: vector search + BM25 keyword search via Azure AI Search.

Also supports:
  - Semantic reranking (Azure AI Search semantic configuration)
  - Metadata filtering (by source file, page, is_table)
  - Returning ranked, deduplicated context chunks
"""

import os
from dataclasses import dataclass
from typing import Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType,
)
from openai import AzureOpenAI


INDEX_NAME = "edi-documents"
TOP_K      = 5      # number of chunks to retrieve
HYBRID     = True   # combine vector + keyword


@dataclass
class RetrievedChunk:
    chunk_id: str
    source:   str
    section:  str
    page:     int
    content:  str
    score:    float
    is_table: bool


def _get_clients():
    endpoint    = os.environ["AZURE_SEARCH_ENDPOINT"]
    key         = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    aoai_ep     = os.environ["AZURE_OPENAI_ENDPOINT"]
    aoai_key    = os.environ["AZURE_OPENAI_API_KEY"]
    embed_deploy= os.environ.get("AZURE_OPENAI_EMBED_DEPLOY", "text-embedding-3-small")

    search  = SearchClient(endpoint, INDEX_NAME, AzureKeyCredential(key))
    openai  = AzureOpenAI(azure_endpoint=aoai_ep, api_key=aoai_key, api_version="2024-02-01")
    return search, openai, embed_deploy


# Module-level cached clients (reuse across requests)
_search_client:  Optional[SearchClient] = None
_openai_client:  Optional[AzureOpenAI]  = None
_embed_deploy:   Optional[str]          = None


def _init():
    global _search_client, _openai_client, _embed_deploy
    if _search_client is None:
        _search_client, _openai_client, _embed_deploy = _get_clients()


def embed_query(query: str) -> list[float]:
    _init()
    response = _openai_client.embeddings.create(input=[query], model=_embed_deploy)
    return response.data[0].embedding


def retrieve(
    query: str,
    top_k: int = TOP_K,
    filter_source: Optional[str] = None,
    filter_tables_only: bool = False,
) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query:              Natural language question
        top_k:              Number of results
        filter_source:      Restrict to a specific source filename
        filter_tables_only: Only return table chunks

    Returns:
        List of RetrievedChunk sorted by relevance
    """
    _init()

    # Build OData filter
    filters = []
    if filter_source:
        filters.append(f"source eq '{filter_source}'")
    if filter_tables_only:
        filters.append("is_table eq true")
    odata_filter = " and ".join(filters) if filters else None

    # Embed the query
    query_vector = embed_query(query)

    vector_query = VectorizedQuery(
        vector       = query_vector,
        k_nearest_neighbors = top_k * 2,   # over-fetch, then rerank
        fields       = "content_vector",
    )

    search_kwargs = dict(
        search_text    = query if HYBRID else None,   # None = pure vector
        vector_queries = [vector_query],
        select         = ["chunk_id", "content", "source", "section", "page", "is_table"],
        filter         = odata_filter,
        top            = top_k,
        query_type     = QueryType.SEMANTIC if _semantic_available() else QueryType.SIMPLE,
    )

    if _semantic_available():
        search_kwargs.update(dict(
            semantic_configuration_name = "default",
            query_caption  = QueryCaptionType.EXTRACTIVE,
            query_answer   = QueryAnswerType.EXTRACTIVE,
        ))

    results = _search_client.search(**search_kwargs)

    chunks: list[RetrievedChunk] = []
    for r in results:
        chunks.append(RetrievedChunk(
            chunk_id = r["chunk_id"],
            source   = r.get("source", ""),
            section  = r.get("section", ""),
            page     = r.get("page", 0),
            content  = r["content"],
            score    = r.get("@search.score", 0.0),
            is_table = r.get("is_table", False),
        ))

    return chunks


def _semantic_available() -> bool:
    """Check if semantic search tier is available (S1+ required)."""
    return os.environ.get("AZURE_SEARCH_SEMANTIC", "false").lower() == "true"


def format_context(chunks: list[RetrievedChunk], max_chars: int = 6000) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.
    Prioritizes higher-scored chunks and respects a character budget.
    """
    parts  = []
    used   = 0
    for c in chunks:
        header = f"[Source: {c.source} | Section: {c.section} | Page: {c.page}]\n"
        block  = header + c.content + "\n\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "".join(parts).strip()


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is the drop reason code for customer returning to utility?"
    results = retrieve(q)
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score={r.score:.4f}) ---")
        print(f"Source:  {r.source}")
        print(f"Section: {r.section}")
        print(f"Content: {r.content[:300]}...")
