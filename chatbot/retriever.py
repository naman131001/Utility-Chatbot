"""
Hybrid Retriever — Azure AI Search

Combines three retrieval strategies in one query:
  1. BM25 keyword search       — exact term matching
  2. Vector search             — semantic similarity via embeddings
  3. Semantic re-ranking       — Azure's L2 re-ranker for final ordering

EDI-aware features:
  - Detects EDI segment codes in the query (e.g. REF*7G) and applies a filter boost
  - Detects version references and can filter to a specific version
  - Returns structured results with chunk metadata for citation in chatbot responses
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType,
    SemanticConfiguration,
)
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from config.settings import search_cfg, openai_cfg

logger = logging.getLogger(__name__)

EDI_CODE_RE = re.compile(
    r'\b(REF|DTM|N1|N3|N4|LIN|ASI|PER|BGN|ST|SE)[*^]([A-Z0-9]{1,3})\b'
)
VERSION_RE = re.compile(r'\bv?(\d+\.\d+)\b')


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    score: float
    source_file: str
    page_number: int
    section_title: str
    segment_codes: list[str]
    version_refs: list[str]
    source_url: str
    captions: list[str]         # Semantic highlight snippets


class HybridRetriever:
    """
    Performs hybrid (keyword + vector + semantic rerank) search
    against the EDI 814 Azure AI Search index.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.search_client = SearchClient(
            endpoint=search_cfg.endpoint,
            index_name=search_cfg.index_name,
            credential=AzureKeyCredential(search_cfg.api_key),
        )
        self.openai_client = AzureOpenAI(
            azure_endpoint=openai_cfg.endpoint,
            api_key=openai_cfg.api_key,
            api_version=openai_cfg.api_version,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_version: Optional[str] = None,
        filter_page: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Run a hybrid search and return ranked chunks.

        Args:
            query          : Natural language or EDI code query.
            top_k          : Override default result count.
            filter_version : Restrict to specific doc version (e.g. "1.6").
            filter_page    : Restrict to a specific page number.
        """
        k = top_k or self.top_k

        # Build OData filter string
        filters = self._build_filter(query, filter_version, filter_page)

        # Generate query embedding
        query_vector = self._embed_query(query)

        # Build the vector query object
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=k * 2,      # Over-fetch, then re-rank
            fields="content_vector",
        )

        # Execute hybrid search with semantic re-ranking
        results = self.search_client.search(
            search_text=query,              # BM25 component
            vector_queries=[vector_query],  # Vector component
            filter=filters,
            top=k,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="edi-semantic-config",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            select=[
                "id", "content", "source_file", "page_number",
                "section_title", "segment_codes", "version_refs", "source_url"
            ],
            scoring_profile="edi-boost",
        )

        return [self._to_result(r) for r in results]

    def _embed_query(self, query: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            input=query,
            model=openai_cfg.embedding_deployment,
        )
        return response.data[0].embedding

    def _build_filter(
        self,
        query: str,
        filter_version: Optional[str],
        filter_page: Optional[int],
    ) -> Optional[str]:
        clauses = []

        if filter_version:
            clauses.append(f"version_refs/any(v: v eq '{filter_version}')")

        if filter_page:
            clauses.append(f"page_number eq {filter_page}")

        return " and ".join(clauses) if clauses else None

    def _to_result(self, raw) -> SearchResult:
        captions = []
        if hasattr(raw, "@search.captions") and raw["@search.captions"]:
            captions = [c.text for c in raw["@search.captions"] if c.text]

        return SearchResult(
            chunk_id=raw.get("id", ""),
            content=raw.get("content", ""),
            score=raw.get("@search.reranker_score") or raw.get("@search.score", 0.0),
            source_file=raw.get("source_file", ""),
            page_number=raw.get("page_number", 0),
            section_title=raw.get("section_title", ""),
            segment_codes=raw.get("segment_codes") or [],
            version_refs=raw.get("version_refs") or [],
            source_url=raw.get("source_url", ""),
            captions=captions,
        )
