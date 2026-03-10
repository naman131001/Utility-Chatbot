"""
Azure AI Search — Skillset Builder

Defines a multi-stage cognitive skillset that enriches EDI documents during indexing:

  Skill 1 — SplitSkill       : Split raw text into pages (token-aware)
  Skill 2 — EntityRecognition : Extract org names, dates, locations
  Skill 3 — KeyPhraseExtraction: Pull key phrases from each chunk
  Skill 4 — CustomWebApiSkill : Call our local EDI entity extractor
                                 (extracts segment codes, element codes)
  Skill 5 — AzureOpenAIEmbeddingSkill: Generate embeddings per chunk

Note: This skillset works alongside the indexer (for blob-triggered pipelines).
For direct push ingestion (pipeline/ingestion.py), the skillset is optional.
"""

import json
import logging
import requests

from config.settings import search_cfg, openai_cfg, blob_cfg

logger = logging.getLogger(__name__)

HEADERS = {
    "Content-Type": "application/json",
    "api-key": search_cfg.api_key,
}

SKILLSET_URL = (
    f"{search_cfg.endpoint}/skillsets/{search_cfg.skillset_name}"
    f"?api-version={search_cfg.api_version}"
)


def build_skillset(custom_skill_url: str = "") -> dict:
    """
    Build the skillset definition.

    Args:
        custom_skill_url: HTTPS endpoint of the Azure Function implementing
                          the custom EDI entity extractor. Leave empty to skip.
    """
    skills = [

        # ── Skill 1: Split text into manageable pages ─────────────────────────
        {
            "@odata.type": "#Microsoft.Skills.Text.SplitSkill",
            "name": "split-skill",
            "description": "Split document text into pages for downstream skills",
            "textSplitMode": "pages",
            "maximumPageLength": 2000,
            "pageOverlapLength": 200,
            "context": "/document",
            "inputs": [
                {"name": "text", "source": "/document/content"}
            ],
            "outputs": [
                {"name": "textItems", "targetName": "pages"}
            ],
        },

        # ── Skill 2: Entity Recognition ───────────────────────────────────────
        {
            "@odata.type": "#Microsoft.Skills.Text.V3.EntityRecognitionSkill",
            "name": "entity-recognition",
            "description": "Extract organizations, dates, and locations",
            "categories": ["Organization", "DateTime", "Location"],
            "defaultLanguageCode": "en",
            "context": "/document/pages/*",
            "inputs": [
                {"name": "text", "source": "/document/pages/*"}
            ],
            "outputs": [
                {"name": "organizations", "targetName": "organizations"},
                {"name": "dateTimes",     "targetName": "dateTimes"},
                {"name": "locations",     "targetName": "locations"},
            ],
        },

        # ── Skill 3: Key Phrase Extraction ────────────────────────────────────
        {
            "@odata.type": "#Microsoft.Skills.Text.KeyPhraseExtractionSkill",
            "name": "key-phrase-extraction",
            "description": "Extract key phrases to improve search relevance",
            "defaultLanguageCode": "en",
            "maxKeyPhraseCount": 10,
            "context": "/document/pages/*",
            "inputs": [
                {"name": "text", "source": "/document/pages/*"}
            ],
            "outputs": [
                {"name": "keyPhrases", "targetName": "keyPhrases"}
            ],
        },

        # ── Skill 4: Azure OpenAI Embedding ───────────────────────────────────
        {
            "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
            "name": "embedding-skill",
            "description": "Generate vector embeddings for semantic search",
            "context": "/document/pages/*",
            "resourceUri": openai_cfg.endpoint,
            "apiKey": openai_cfg.api_key,
            "deploymentId": openai_cfg.embedding_deployment,
            "modelName": openai_cfg.embedding_deployment,
            "inputs": [
                {"name": "text", "source": "/document/pages/*"}
            ],
            "outputs": [
                {"name": "embedding", "targetName": "content_vector"}
            ],
        },
    ]

    # ── Skill 5: Custom EDI Entity Extractor (optional Azure Function) ────────
    if custom_skill_url:
        skills.append({
            "@odata.type": "#Microsoft.Skills.Custom.WebApiSkill",
            "name": "edi-entity-extractor",
            "description": "Custom skill: extract EDI segment codes and element codes",
            "uri": custom_skill_url,
            "httpMethod": "POST",
            "timeout": "PT30S",
            "batchSize": 4,
            "context": "/document/pages/*",
            "inputs": [
                {"name": "text", "source": "/document/pages/*"}
            ],
            "outputs": [
                {"name": "segment_codes", "targetName": "segment_codes"},
                {"name": "version_refs",  "targetName": "version_refs"},
            ],
            "httpHeaders": {},
        })

    return {
        "name": search_cfg.skillset_name,
        "description": "EDI 814 document enrichment skillset",
        "skills": skills,

        # ── Knowledge store (optional — writes enriched data to blob/table) ───
        "knowledgeStore": {
            "storageConnectionString": blob_cfg.connection_string,
            "projections": [
                {
                    "tables": [
                        {
                            "tableName": "edi814Chunks",
                            "generatedKeyName": "chunkId",
                            "source": "/document/pages/*",
                            "inputs": [
                                {"name": "content",       "source": "/document/pages/*"},
                                {"name": "keyPhrases",    "source": "/document/pages/*/keyPhrases"},
                                {"name": "organizations", "source": "/document/pages/*/organizations"},
                            ],
                        }
                    ],
                    "objects": [],
                    "files": [],
                }
            ],
        },

        # ── Index projections (map enriched fields → index fields) ─────────────
        "indexProjections": {
            "selectors": [
                {
                    "targetIndexName": search_cfg.index_name,
                    "parentKeyFieldName": "parent_id",
                    "sourceContext": "/document/pages/*",
                    "mappings": [
                        {"name": "content",        "source": "/document/pages/*"},
                        {"name": "content_vector", "source": "/document/pages/*/content_vector"},
                        {"name": "source_file",    "source": "/document/metadata_storage_name"},
                        {"name": "source_url",     "source": "/document/metadata_storage_path"},
                    ],
                }
            ],
            "parameters": {"projectionMode": "generatedKeyAsId"},
        },
    }


class SkillsetManager:
    """Creates, updates, and deletes the Azure AI Search skillset."""

    def create_or_update(self, custom_skill_url: str = "") -> dict:
        schema = build_skillset(custom_skill_url)
        response = requests.put(SKILLSET_URL, headers=HEADERS, json=schema)
        response.raise_for_status()
        logger.info(f"Skillset '{search_cfg.skillset_name}' created/updated.")
        return response.json()

    def delete(self) -> None:
        response = requests.delete(SKILLSET_URL, headers=HEADERS)
        if response.status_code != 404:
            response.raise_for_status()
        logger.info(f"Skillset '{search_cfg.skillset_name}' deleted.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    custom_url = sys.argv[1] if len(sys.argv) > 1 else ""
    manager = SkillsetManager()
    result = manager.create_or_update(custom_skill_url=custom_url)
    print(json.dumps(result, indent=2))
