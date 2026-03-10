"""
Azure AI Search — Index Schema Builder

Creates the search index with:
  - Vector field  : semantic / similarity search via embeddings
  - Text fields   : BM25 keyword search on content + section title
  - Filter fields : page, content_type, segment_codes, version_refs
  - Hybrid search : combines BM25 + vector by default

Run once to provision the index, or call create_or_update() to safely re-apply.
"""

import logging
import requests

from config.settings import search_cfg, openai_cfg

logger = logging.getLogger(__name__)

HEADERS = {
    "Content-Type": "application/json",
    "api-key": search_cfg.api_key,
}


def build_index_schema() -> dict:
    """Return the full index definition as a dict."""
    return {
        "name": search_cfg.index_name,
        "fields": [
            # ── Key ───────────────────────────────────────────────────────────
            {
                "name": "id",
                "type": "Edm.String",
                "key": True,
                "filterable": True,
            },

            # ── Primary searchable content ────────────────────────────────────
            {
                "name": "content",
                "type": "Edm.String",
                "searchable": True,
                "retrievable": True,
                "analyzer": "en.microsoft",     # English linguistic analysis
            },
            {
                "name": "section_title",
                "type": "Edm.String",
                "searchable": True,
                "retrievable": True,
                "analyzer": "en.microsoft",
            },

            # ── Vector field (ada-002 = 1536 dims) ───────────────────────────
            {
                "name": "content_vector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "retrievable": False,           # Don't return raw vectors to client
                "dimensions": openai_cfg.embedding_dimensions,
                "vectorSearchProfile": "edi-vector-profile",
            },

            # ── Metadata / filter fields ──────────────────────────────────────
            {
                "name": "source_file",
                "type": "Edm.String",
                "filterable": True,
                "retrievable": True,
                "facetable": True,
            },
            {
                "name": "page_number",
                "type": "Edm.Int32",
                "filterable": True,
                "sortable": True,
                "retrievable": True,
            },
            {
                "name": "content_type",
                "type": "Edm.String",
                "filterable": True,
                "facetable": True,
                "retrievable": True,
            },
            {
                "name": "chunk_index",
                "type": "Edm.Int32",
                "filterable": True,
                "sortable": True,
                "retrievable": True,
            },
            {
                "name": "token_count",
                "type": "Edm.Int32",
                "filterable": True,
                "retrievable": True,
            },

            # ── EDI-specific multi-value fields ───────────────────────────────
            {
                "name": "segment_codes",
                "type": "Collection(Edm.String)",
                "searchable": True,
                "filterable": True,
                "facetable": True,
                "retrievable": True,
            },
            {
                "name": "version_refs",
                "type": "Collection(Edm.String)",
                "filterable": True,
                "facetable": True,
                "retrievable": True,
            },

            # ── Citation URL ──────────────────────────────────────────────────
            {
                "name": "source_url",
                "type": "Edm.String",
                "retrievable": True,
            },
        ],

        # ── Vector search configuration ────────────────────────────────────────
        "vectorSearch": {
            "algorithms": [
                {
                    "name": "edi-hnsw",
                    "kind": "hnsw",
                    "hnswParameters": {
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine",
                    },
                }
            ],
            "profiles": [
                {
                    "name": "edi-vector-profile",
                    "algorithm": "edi-hnsw",
                }
            ],
        },

        # ── Semantic search configuration (re-ranking) ─────────────────────────
        "semantic": {
            "configurations": [
                {
                    "name": "edi-semantic-config",
                    "prioritizedFields": {
                        "titleField": {"fieldName": "section_title"},
                        "contentFields": [{"fieldName": "content"}],
                        "keywordsFields": [{"fieldName": "segment_codes"}],
                    },
                }
            ]
        },

        # ── Scoring profile (boost exact EDI code matches) ─────────────────────
        "scoringProfiles": [
            {
                "name": "edi-boost",
                "text": {
                    "weights": {
                        "content": 1.0,
                        "section_title": 2.0,
                        "segment_codes": 3.0,    # Exact code matches rank higher
                    }
                },
            }
        ],
        "defaultScoringProfile": "edi-boost",
    }


class IndexManager:
    """Creates, updates, and deletes the Azure AI Search index."""

    def __init__(self):
        self.base_url = (
            f"{search_cfg.endpoint}/indexes"
            f"?api-version={search_cfg.api_version}"
        )
        self.index_url = (
            f"{search_cfg.endpoint}/indexes/{search_cfg.index_name}"
            f"?api-version={search_cfg.api_version}"
        )

    def create_or_update(self) -> dict:
        """PUT the index schema (creates if not exists, updates if it does)."""
        schema = build_index_schema()
        response = requests.put(
            self.index_url,
            headers=HEADERS,
            json=schema,
        )
        response.raise_for_status()
        logger.info(f"Index '{search_cfg.index_name}' created/updated successfully.")
        return response.json()

    def delete(self) -> None:
        """Delete the index. WARNING: destroys all indexed data."""
        response = requests.delete(self.index_url, headers=HEADERS)
        if response.status_code == 404:
            logger.warning("Index not found — nothing to delete.")
        else:
            response.raise_for_status()
            logger.info(f"Index '{search_cfg.index_name}' deleted.")

    def stats(self) -> dict:
        """Return document count and storage size."""
        url = (
            f"{search_cfg.endpoint}/indexes/{search_cfg.index_name}/stats"
            f"?api-version={search_cfg.api_version}"
        )
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    manager = IndexManager()
    result = manager.create_or_update()
    print(json.dumps(result, indent=2))
    stats = manager.stats()
    print(f"\nIndex stats: {json.dumps(stats, indent=2)}")
