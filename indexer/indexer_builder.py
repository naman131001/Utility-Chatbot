"""
Azure AI Search — Indexer Builder

The indexer connects:
    Blob Storage (data source)  →  Skillset  →  Search Index

It runs on a schedule OR can be triggered manually.
Supports:
  - Change detection (only re-index modified blobs)
  - Deletion detection (remove docs when blob is deleted)
  - Field mappings from blob metadata → index fields
"""

import json
import logging
import requests

from config.settings import search_cfg, blob_cfg

logger = logging.getLogger(__name__)

HEADERS = {
    "Content-Type": "application/json",
    "api-key": search_cfg.api_key,
}


class DataSourceManager:
    """Creates the blob storage data source for the indexer."""

    def create_or_update(self) -> dict:
        url = (
            f"{search_cfg.endpoint}/datasources/{search_cfg.datasource_name}"
            f"?api-version={search_cfg.api_version}"
        )
        schema = {
            "name": search_cfg.datasource_name,
            "description": "EDI 814 documents in Azure Blob Storage",
            "type": "azureblob",
            "credentials": {
                "connectionString": blob_cfg.connection_string
            },
            "container": {
                "name": blob_cfg.container_name,
                "query": None,          # Index all blobs; use "subfolder/" to limit
            },
            # Re-index if blob's LastModified changes
            "dataChangeDetectionPolicy": {
                "@odata.type": (
                    "#Microsoft.Azure.Search"
                    ".HighWaterMarkChangeDetectionPolicy"
                ),
                "highWaterMarkColumnName": "metadata_storage_last_modified",
            },
            # Remove index docs when blob is deleted
            "dataDeletionDetectionPolicy": {
                "@odata.type": (
                    "#Microsoft.Azure.Search"
                    ".SoftDeleteColumnDeletionDetectionPolicy"
                ),
                "softDeleteColumnName":  "IsDeleted",
                "softDeleteMarkerValue": "true",
            },
        }

        response = requests.put(url, headers=HEADERS, json=schema)
        response.raise_for_status()
        logger.info(f"Data source '{search_cfg.datasource_name}' created/updated.")
        return response.json()


class IndexerManager:
    """Creates, runs, and monitors the Azure AI Search indexer."""

    def __init__(self):
        self.indexer_url = (
            f"{search_cfg.endpoint}/indexers/{search_cfg.indexer_name}"
            f"?api-version={search_cfg.api_version}"
        )
        self.run_url = (
            f"{search_cfg.endpoint}/indexers/{search_cfg.indexer_name}/run"
            f"?api-version={search_cfg.api_version}"
        )
        self.status_url = (
            f"{search_cfg.endpoint}/indexers/{search_cfg.indexer_name}/status"
            f"?api-version={search_cfg.api_version}"
        )

    def create_or_update(self) -> dict:
        schema = {
            "name": search_cfg.indexer_name,
            "description": "EDI 814 blob indexer with skillset enrichment",
            "dataSourceName": search_cfg.datasource_name,
            "targetIndexName": search_cfg.index_name,
            "skillsetName":    search_cfg.skillset_name,

            # ── Schedule: run every 2 hours ────────────────────────────────────
            "schedule": {
                "interval": "PT2H",         # ISO 8601 duration
                "startTime": "2024-01-01T00:00:00Z",
            },

            # ── Parameters ─────────────────────────────────────────────────────
            "parameters": {
                "batchSize": 10,            # Documents per batch
                "maxFailedItems": 5,        # Tolerate up to 5 failures
                "maxFailedItemsPerBatch": 2,
                "configuration": {
                    # Parse markdown as text
                    "parsingMode": "default",
                    # Include blob metadata as index fields
                    "indexStorageMetadataOnTimeoutBlobs": True,
                    "indexedFileNameExtensions": ".md,.txt,.pdf",
                    "excludedFileNameExtensions": ".png,.jpg,.gif",
                },
            },

            # ── Field mappings (blob metadata → index fields) ──────────────────
            "fieldMappings": [
                {
                    "sourceFieldName": "metadata_storage_path",
                    "targetFieldName": "id",
                    "mappingFunction": {
                        "name": "base64Encode"          # Keys must be URL-safe
                    },
                },
                {
                    "sourceFieldName": "metadata_storage_name",
                    "targetFieldName": "source_file",
                },
                {
                    "sourceFieldName": "metadata_storage_path",
                    "targetFieldName": "source_url",
                },
            ],

            # ── Output field mappings (skillset outputs → index fields) ─────────
            "outputFieldMappings": [
                {
                    "sourceFieldName": "/document/pages/*/content_vector",
                    "targetFieldName": "content_vector",
                },
                {
                    "sourceFieldName": "/document/pages/*/keyPhrases",
                    "targetFieldName": "key_phrases",
                },
                {
                    "sourceFieldName": "/document/pages/*/organizations",
                    "targetFieldName": "organizations",
                },
            ],
        }

        response = requests.put(self.indexer_url, headers=HEADERS, json=schema)
        response.raise_for_status()
        logger.info(f"Indexer '{search_cfg.indexer_name}' created/updated.")
        return response.json()

    def run(self) -> None:
        """Trigger an immediate indexer run."""
        response = requests.post(self.run_url, headers=HEADERS)
        response.raise_for_status()
        logger.info(f"Indexer '{search_cfg.indexer_name}' run triggered.")

    def status(self) -> dict:
        """Return last execution status."""
        response = requests.get(self.status_url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        last = data.get("lastResult", {})
        return {
            "status":          data.get("status"),
            "lastRun":         last.get("endTime"),
            "itemsProcessed":  last.get("itemsProcessed"),
            "itemsFailed":     last.get("itemsFailed"),
            "errors":          last.get("errors", []),
        }

    def reset(self) -> None:
        """Reset the indexer (forces full re-index on next run)."""
        reset_url = (
            f"{search_cfg.endpoint}/indexers/{search_cfg.indexer_name}/reset"
            f"?api-version={search_cfg.api_version}"
        )
        response = requests.post(reset_url, headers=HEADERS)
        response.raise_for_status()
        logger.info(f"Indexer '{search_cfg.indexer_name}' reset.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    action = sys.argv[1] if len(sys.argv) > 1 else "status"

    ds = DataSourceManager()
    idx = IndexerManager()

    if action == "create":
        ds.create_or_update()
        idx.create_or_update()
        print("✅ Data source and indexer created.")

    elif action == "run":
        idx.run()
        print("▶️  Indexer run triggered.")

    elif action == "status":
        print(json.dumps(idx.status(), indent=2))

    elif action == "reset":
        idx.reset()
        print("🔄 Indexer reset — will re-index all docs on next run.")

    else:
        print(f"Unknown action: {action}")
        print("Usage: python -m indexer.indexer_builder [create|run|status|reset]")
