"""
indexer.py
----------
Creates the Azure AI Search index (with vector field) and indexes all chunks.

Requires env vars:
  AZURE_SEARCH_ENDPOINT     e.g. https://my-search.search.windows.net
  AZURE_SEARCH_ADMIN_KEY
  AZURE_OPENAI_ENDPOINT     e.g. https://my-aoai.openai.azure.com
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_EMBED_DEPLOY e.g. text-embedding-3-small  (deployment name)
"""

import os
import time
import json
import asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchIndex,
)
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

from chunker_2 import Chunk, chunk_directory


# ── Config ────────────────────────────────────────────────────────────────────

INDEX_NAME        = "edi-documents"
VECTOR_DIMENSIONS = 1536           # ada-002 or text-embedding-3-small
BATCH_SIZE        = 50             # documents per upload batch
EMBED_BATCH_SIZE  = 4              # S0 tier is very limited — keep this small
EMBED_SLEEP_SEC   = 2.0            # pause between every embedding call
EMBED_MAX_RETRIES = 6              # max retries on 429


def _get_clients():
    search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    search_key      = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    aoai_endpoint   = os.environ["AZURE_OPENAI_ENDPOINT"]
    aoai_key        = os.environ["AZURE_OPENAI_API_KEY"]
    embed_deploy    = os.environ.get("AZURE_OPENAI_EMBED_DEPLOY", "text-embedding-3-small")

    index_client  = SearchIndexClient(search_endpoint, AzureKeyCredential(search_key))
    search_client = SearchClient(search_endpoint, INDEX_NAME, AzureKeyCredential(search_key))
    openai_client = AzureOpenAI(azure_endpoint=aoai_endpoint, api_key=aoai_key, api_version="2024-02-01")

    return index_client, search_client, openai_client, embed_deploy


# ── Index schema ──────────────────────────────────────────────────────────────

def create_index(index_client: SearchIndexClient) -> None:
    """Create or update the Azure AI Search index."""

    fields = [
        SimpleField(
            name="chunk_id", type=SearchFieldDataType.String,
            key=True, filterable=True
        ),
        SearchableField(
            name="content", type=SearchFieldDataType.String,
            analyzer_name="en.microsoft"
        ),
        SimpleField(
            name="source", type=SearchFieldDataType.String,
            filterable=True, facetable=True
        ),
        SimpleField(
            name="section", type=SearchFieldDataType.String,
            filterable=True, searchable=True
        ),
        SimpleField(
            name="page", type=SearchFieldDataType.Int32,
            filterable=True, sortable=True
        ),
        SimpleField(
            name="is_table", type=SearchFieldDataType.Boolean,
            filterable=True
        ),
        SimpleField(
            name="metadata_json", type=SearchFieldDataType.String,
        ),
        # Vector field for semantic/ANN search
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="my-hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="my-hnsw-config")
        ],
        profiles=[
            VectorSearchProfile(
                name="my-hnsw-profile",
                algorithm_configuration_name="my-hnsw-config"
            )
        ],
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )

    result = index_client.create_or_update_index(index)
    print(f"✅ Index '{result.name}' created/updated.")


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], client: AzureOpenAI, deployment: str) -> list[list[float]]:
    """
    Call Azure OpenAI Embeddings in small batches with exponential backoff.
    Safe for S0 tier (low TPM quota).
    """
    import re as _re
    all_embeddings = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]

        for attempt in range(EMBED_MAX_RETRIES):
            try:
                response = client.embeddings.create(input=batch, model=deployment)
                all_embeddings.extend([item.embedding for item in response.data])
                done = min(i + EMBED_BATCH_SIZE, len(texts))
                print(f"    Embedded {done}/{len(texts)} chunks ...", end="\r")
                time.sleep(EMBED_SLEEP_SEC)
                break

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RateLimitReached" in err_str:
                    match = _re.search(r'retry after (\d+) second', err_str, _re.IGNORECASE)
                    wait = int(match.group(1)) if match else (15 * (2 ** attempt))
                    print(f"\n    ⚠️  Rate limited. Waiting {wait}s (retry {attempt+1}/{EMBED_MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Embedding failed after {EMBED_MAX_RETRIES} retries for batch at index {i}")

    print()
    return all_embeddings


# ── Upserting ─────────────────────────────────────────────────────────────────

def upload_chunks(
    chunks: list[Chunk],
    search_client: SearchClient,
    openai_client: AzureOpenAI,
    embed_deploy: str,
) -> None:
    """Embed and upload chunks to Azure AI Search in batches."""

    total = len(chunks)
    print(f"Uploading {total} chunks ...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        # Embed the content of each chunk
        texts      = [c.content for c in batch]
        embeddings = embed_texts(texts, openai_client, embed_deploy)

        docs = []
        for chunk, vec in zip(batch, embeddings):
            docs.append({
                "chunk_id":       chunk.chunk_id,
                "content":        chunk.content,
                "source":         chunk.source,
                "section":        chunk.section,
                "page":           chunk.page or 0,
                "is_table":       chunk.is_table,
                "metadata_json":  json.dumps(chunk.metadata),
                "content_vector": vec,
            })

        result = search_client.upload_documents(documents=docs)
        succeeded = sum(1 for r in result if r.succeeded)
        print(f"  Batch {batch_start // BATCH_SIZE + 1}: {succeeded}/{len(docs)} uploaded")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_indexer(documents_folder: str) -> None:
    index_client, search_client, openai_client, embed_deploy = _get_clients()

    # 1. Create / update index schema
    create_index(index_client)

    # 2. Chunk all markdown files
    chunks = chunk_directory(documents_folder)

    if not chunks:
        print("No chunks found. Check your documents folder.")
        return

    # 3. Upload with embeddings
    upload_chunks(chunks, search_client, openai_client, embed_deploy)
    print(f"\n✅ Indexing complete. {len(chunks)} chunks in index '{INDEX_NAME}'.")


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "./documents"
    run_indexer(folder)