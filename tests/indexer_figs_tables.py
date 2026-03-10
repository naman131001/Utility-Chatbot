"""
indexer.py  — Azure AI Search Indexer v3
-----------------------------------------
Creates / updates the Azure AI Search index schema to match the enriched
Chunk dataclass produced by chunker.py v3, then embeds and uploads all chunks.

Environment variables required
================================
  AZURE_SEARCH_ENDPOINT       e.g. https://my-search.search.windows.net
  AZURE_SEARCH_ADMIN_KEY
  AZURE_OPENAI_ENDPOINT       e.g. https://my-aoai.openai.azure.com
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_EMBED_DEPLOY   deployment name, e.g. text-embedding-3-small

Optional
========
  SEARCH_INDEX_NAME           default: "edi-documents"
  VECTOR_DIMENSIONS           default: 1536 (ada-002 / text-embedding-3-small)
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from openai import AzureOpenAI

from chunker import Chunk, chunk_directory

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

INDEX_NAME        = os.environ.get("SEARCH_INDEX_NAME", "edi-documents")
VECTOR_DIMENSIONS = int(os.environ.get("VECTOR_DIMENSIONS", "1536"))
UPLOAD_BATCH_SIZE = 50    # documents per upload call
EMBED_BATCH_SIZE  = 4     # texts per embedding API call  (S0 tier safe)
EMBED_SLEEP_SEC   = 2.0   # pause between embedding batches
EMBED_MAX_RETRIES = 6


# ─────────────────────────────────────────────────────────────────────────────
# Client factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_clients():
    search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    search_key      = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    aoai_endpoint   = os.environ["AZURE_OPENAI_ENDPOINT"]
    aoai_key        = os.environ["AZURE_OPENAI_API_KEY"]
    embed_deploy    = os.environ.get("AZURE_OPENAI_EMBED_DEPLOY",
                                     "text-embedding-3-small")

    cred          = AzureKeyCredential(search_key)
    index_client  = SearchIndexClient(search_endpoint, cred)
    search_client = SearchClient(search_endpoint, INDEX_NAME, cred)
    openai_client = AzureOpenAI(
        azure_endpoint=aoai_endpoint,
        api_key=aoai_key,
        api_version="2024-02-01",
    )
    return index_client, search_client, openai_client, embed_deploy


# ─────────────────────────────────────────────────────────────────────────────
# Index schema
# ─────────────────────────────────────────────────────────────────────────────

def create_index(index_client: SearchIndexClient) -> None:
    """Create or update the Azure AI Search index with the v3 schema."""

    fields = [
        # ── Identity ──────────────────────────────────────────────────────
        SimpleField(
            name="chunk_id", type=SearchFieldDataType.String,
            key=True, filterable=True,
        ),
        SimpleField(
            name="source", type=SearchFieldDataType.String,
            filterable=True, facetable=True,
        ),
        # Full URL to the original PDF — returned with every result
        SimpleField(
            name="source_pdf_url", type=SearchFieldDataType.String,
            filterable=True,
        ),
        # Human-readable filename of the PDF
        SimpleField(
            name="source_pdf_name", type=SearchFieldDataType.String,
            filterable=True, facetable=True,
        ),
        SimpleField(
            name="chunk_index", type=SearchFieldDataType.Int32,
            filterable=True, sortable=True,
        ),

        # ── Location ──────────────────────────────────────────────────────
        SimpleField(
            name="page_start", type=SearchFieldDataType.Int32,
            filterable=True, sortable=True,
        ),
        SimpleField(
            name="page_end", type=SearchFieldDataType.Int32,
            filterable=True, sortable=True,
        ),

        # ── Hierarchy / navigation ────────────────────────────────────────
        SimpleField(
            name="heading_level", type=SearchFieldDataType.Int32,
            filterable=True, sortable=True,
        ),
        SearchableField(
            name="topic", type=SearchFieldDataType.String,
            filterable=True, facetable=True,
            analyzer_name="en.microsoft",
        ),
        SearchableField(
            name="subtopic", type=SearchFieldDataType.String,
            filterable=True, facetable=True,
            analyzer_name="en.microsoft",
        ),
        SearchableField(
            name="section_title", type=SearchFieldDataType.String,
            filterable=True,
            analyzer_name="en.microsoft",
        ),
        SearchableField(
            name="section", type=SearchFieldDataType.String,
            filterable=True,
            analyzer_name="en.microsoft",
        ),

        # ── Content ───────────────────────────────────────────────────────
        SearchableField(
            name="content", type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),

        # ── Content-type flags ────────────────────────────────────────────
        SimpleField(
            name="content_type", type=SearchFieldDataType.String,
            filterable=True, facetable=True,
        ),
        SimpleField(
            name="has_table", type=SearchFieldDataType.Boolean,
            filterable=True,
        ),
        SimpleField(
            name="has_figure", type=SearchFieldDataType.Boolean,
            filterable=True,
        ),
        SimpleField(
            name="is_table", type=SearchFieldDataType.Boolean,
            filterable=True,
        ),
        SimpleField(
            name="is_figure", type=SearchFieldDataType.Boolean,
            filterable=True,
        ),
        SimpleField(
            name="table_count", type=SearchFieldDataType.Int32,
            filterable=True,
        ),
        SimpleField(
            name="figure_count", type=SearchFieldDataType.Int32,
            filterable=True,
        ),

        # ── Extra metadata as JSON string ─────────────────────────────────
        SimpleField(
            name="metadata_json", type=SearchFieldDataType.String,
        ),

        # ── Vector field ──────────────────────────────────────────────────
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
        profiles=[VectorSearchProfile(
            name="hnsw-profile",
            algorithm_configuration_name="hnsw-config",
        )],
    )

    # Semantic search configuration — boosts title + content relevance
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="section_title"),
                    keywords_fields=[
                        SemanticField(field_name="topic"),
                        SemanticField(field_name="subtopic"),
                        SemanticField(field_name="section"),
                    ],
                    content_fields=[SemanticField(field_name="content")],
                ),
            )
        ]
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    result = index_client.create_or_update_index(index)
    print(f"✅  Index '{result.name}' created / updated.")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(
    texts:      list[str],
    client:     AzureOpenAI,
    deployment: str,
) -> list[list[float]]:
    """Embed texts in small batches with exponential-back-off on 429s."""
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]

        for attempt in range(EMBED_MAX_RETRIES):
            try:
                response = client.embeddings.create(
                    input=batch, model=deployment
                )
                all_embeddings.extend(item.embedding for item in response.data)
                done = min(i + EMBED_BATCH_SIZE, len(texts))
                print(f"    Embedded {done}/{len(texts)} …", end="\r")
                # time.sleep(EMBED_SLEEP_SEC)
                break

            except Exception as exc:
                err = str(exc)
                if "429" in err or "RateLimitReached" in err:
                    wait_m = re.search(
                        r'retry after (\d+) second', err, re.IGNORECASE
                    )
                    wait = int(wait_m.group(1)) if wait_m else (15 * 2 ** attempt)
                    print(f"\n    ⚠️  Rate-limited. Waiting {wait}s "
                          f"(attempt {attempt + 1}/{EMBED_MAX_RETRIES}) …")
                    # time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(
                f"Embedding failed after {EMBED_MAX_RETRIES} retries "
                f"at batch index {i}"
            )

    print()
    return all_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_to_doc(chunk: Chunk, vector: list[float]) -> dict:
    """Serialise a Chunk to an Azure Search document."""
    return {
        "chunk_id":        chunk.chunk_id,
        "source":          chunk.source,
        "source_pdf_url":  chunk.source_pdf_url,
        "source_pdf_name": chunk.source_pdf_name,
        "chunk_index":    chunk.chunk_index,
        "page_start":     chunk.page_start or 0,
        "page_end":       chunk.page_end   or 0,
        "heading_level":  chunk.heading_level,
        "topic":          chunk.topic,
        "subtopic":       chunk.subtopic,
        "section_title":  chunk.section_title,
        "section":        chunk.section,
        "content":        chunk.content,
        "content_type":   chunk.content_type,
        "has_table":      chunk.has_table,
        "has_figure":     chunk.has_figure,
        "is_table":       chunk.is_table,
        "is_figure":      chunk.is_figure,
        "table_count":    chunk.table_count,
        "figure_count":   chunk.figure_count,
        "metadata_json":  json.dumps(chunk.metadata, ensure_ascii=False),
        "content_vector": vector,
    }


def upload_chunks(
    chunks:        list[Chunk],
    search_client: SearchClient,
    openai_client: AzureOpenAI,
    embed_deploy:  str,
) -> None:
    total = len(chunks)
    print(f"Uploading {total} chunks …")

    for batch_start in range(0, total, UPLOAD_BATCH_SIZE):
        batch   = chunks[batch_start : batch_start + UPLOAD_BATCH_SIZE]
        texts   = [c.content for c in batch]
        vectors = embed_texts(texts, openai_client, embed_deploy)

        docs    = [_chunk_to_doc(c, v) for c, v in zip(batch, vectors)]
        results = search_client.upload_documents(documents=docs)
        ok      = sum(1 for r in results if r.succeeded)
        bn      = batch_start // UPLOAD_BATCH_SIZE + 1
        print(f"  Batch {bn}: {ok}/{len(docs)} succeeded")

    print(f"\n✅  Done — {total} chunks indexed into '{INDEX_NAME}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Index management helpers
# ─────────────────────────────────────────────────────────────────────────────

def delete_index(index_client: SearchIndexClient) -> None:
    """Delete the index if it exists. Required when changing vector algorithm config."""
    try:
        index_client.delete_index(INDEX_NAME)
        print(f"🗑️   Index '{INDEX_NAME}' deleted.")
    except Exception as exc:
        if "ResourceNotFound" in str(exc) or "404" in str(exc):
            print(f"ℹ️   Index '{INDEX_NAME}' did not exist — nothing to delete.")
        else:
            raise


def index_exists(index_client: SearchIndexClient) -> bool:
    try:
        index_client.get_index(INDEX_NAME)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_indexer(documents_folder: str, recreate: bool = False) -> None:
    index_client, search_client, openai_client, embed_deploy = _get_clients()

    # 1. Optionally wipe the index first (needed when vector algorithm config changed)
    if recreate:
        print("♻️   --recreate flag set: dropping existing index …")
        delete_index(index_client)
    elif index_exists(index_client):
        print(f"ℹ️   Index '{INDEX_NAME}' already exists. "
              "Pass --recreate to drop and rebuild it.")

    # 2. Create / refresh index schema
    create_index(index_client)

    # 3. Chunk all markdown files
    # chunks = chunk_directory(documents_folder)

    chunks = chunk_directory(documents_folder, use_cache=True, force=False)
    if not chunks:
        print("⚠️  No chunks produced — check your documents folder.")
        return

    # 4. Print a brief summary
    tables  = sum(1 for c in chunks if c.is_table)
    figures = sum(1 for c in chunks if c.is_figure)
    print(f"   {len(chunks)} total chunks  |  {tables} table chunks  "
          f"|  {figures} figure chunks")

    # 5. Embed + upload
    upload_chunks(chunks, search_client, openai_client, embed_deploy)


if __name__ == "__main__":
    import sys
    # Usage:
    #   python indexer.py                        → index ./documents
    #   python indexer.py ./my_docs              → index ./my_docs
    #   python indexer.py ./my_docs --recreate   → drop index first, then index
    args = sys.argv[1:]
    recreate_flag = "--recreate" in args
    args = [a for a in args if a != "--recreate"]
    folder = args[0] if args else "./documents"
    run_indexer(folder, recreate=recreate_flag)