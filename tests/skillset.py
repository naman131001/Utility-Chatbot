"""
skillset.py
-----------
Defines an Azure AI Search *native* skillset + indexer pipeline.

Use this INSTEAD of indexer.py if you want Azure to:
  - Pull files from Azure Blob Storage automatically
  - Split text natively (SplitSkill)
  - Embed natively (AzureOpenAIEmbeddingSkill)
  - Index everything without custom Python embedding loops

Requires env vars:
  AZURE_SEARCH_ENDPOINT
  AZURE_SEARCH_ADMIN_KEY
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_EMBED_DEPLOY
  AZURE_BLOB_CONNECTION_STRING   (storage account with your markdown files)
  AZURE_BLOB_CONTAINER           (container name)
"""

import os
import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
    SearchIndexer,
    SearchIndexerSkillset,
    SplitSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    AzureOpenAIEmbeddingSkill,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    FieldMapping,
)


INDEX_NAME     = "edi-documents"
SKILLSET_NAME  = "edi-skillset"
INDEXER_NAME   = "edi-indexer"
DATASOURCE_NAME = "edi-blob-datasource"


def _get_indexer_client():
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    key      = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    return SearchIndexerClient(endpoint, AzureKeyCredential(key))


def create_datasource(client: SearchIndexerClient) -> None:
    """Register Azure Blob Storage as the data source."""
    conn_str  = os.environ["AZURE_BLOB_CONNECTION_STRING"]
    container = os.environ["AZURE_BLOB_CONTAINER"]

    ds = SearchIndexerDataSourceConnection(
        name             = DATASOURCE_NAME,
        type             = "azureblob",
        connection_string= conn_str,
        container        = SearchIndexerDataContainer(name=container),
    )
    client.create_or_update_data_source_connection(ds)
    print(f"✅ Datasource '{DATASOURCE_NAME}' registered.")


def create_skillset(client: SearchIndexerClient) -> None:
    """
    Skillset with:
      1. SplitSkill  → chunks document text into pages
      2. AzureOpenAIEmbeddingSkill → embeds each chunk
    """
    aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    embed_deploy  = os.environ.get("AZURE_OPENAI_EMBED_DEPLOY", "text-embedding-3-small")

    split_skill = SplitSkill(
        name                = "split-skill",
        description         = "Split document into chunks of ~512 tokens",
        text_split_mode     = "pages",
        maximum_page_length = 2048,    # characters; tune to your chunk size
        page_overlap_length = 256,
        default_language_code = "en",
        inputs  = [InputFieldMappingEntry(name="text",   source="/document/content")],
        outputs = [OutputFieldMappingEntry(name="textItems", target_name="pages")],
        context = "/document",
    )

    embedding_skill = AzureOpenAIEmbeddingSkill(
        name             = "embedding-skill",
        description      = "Embed each chunk",
        resource_uri     = aoai_endpoint,
        deployment_id    = embed_deploy,
        model_name       = embed_deploy,
        dimensions       = 1536,
        inputs  = [InputFieldMappingEntry(name="text",   source="/document/pages/*")],
        outputs = [OutputFieldMappingEntry(name="embedding", target_name="content_vector")],
        context = "/document/pages/*",
    )

    # Index projection = how enriched docs map back to the index
    index_projection = SearchIndexerIndexProjection(
        selectors=[
            SearchIndexerIndexProjectionSelector(
                target_index_name = INDEX_NAME,
                parent_key_field_name = "parent_id",
                source_context    = "/document/pages/*",
                mappings=[
                    InputFieldMappingEntry(name="content",        source="/document/pages/*"),
                    InputFieldMappingEntry(name="content_vector", source="/document/pages/*/content_vector"),
                    InputFieldMappingEntry(name="source",         source="/document/metadata_storage_name"),
                ],
            )
        ],
        parameters=SearchIndexerIndexProjectionsParameters(
            projection_mode="generatedKeyAsId"
        ),
    )

    skillset = SearchIndexerSkillset(
        name             = SKILLSET_NAME,
        description      = "EDI document enrichment pipeline",
        skills           = [split_skill, embedding_skill],
        index_projection = index_projection,
    )

    client.create_or_update_skillset(skillset)
    print(f"✅ Skillset '{SKILLSET_NAME}' created.")


def create_indexer(client: SearchIndexerClient) -> None:
    """Create the indexer that wires datasource → skillset → index."""
    indexer = SearchIndexer(
        name               = INDEXER_NAME,
        data_source_name   = DATASOURCE_NAME,
        target_index_name  = INDEX_NAME,
        skillset_name      = SKILLSET_NAME,
        field_mappings=[
            FieldMapping(source_field_name="metadata_storage_name", target_field_name="source"),
        ],
        # Run immediately and then check schedule:
        # schedule=IndexingSchedule(interval=timedelta(hours=1))
    )
    client.create_or_update_indexer(indexer)
    print(f"✅ Indexer '{INDEXER_NAME}' created. Running now...")
    client.run_indexer(INDEXER_NAME)


def run_skillset_pipeline() -> None:
    """Full pipeline: datasource → skillset → indexer."""
    client = _get_indexer_client()
    create_datasource(client)
    create_skillset(client)
    create_indexer(client)
    print("\n✅ Azure-native skillset pipeline is live.")
    print("   Monitor indexer status at: Azure Portal → Search Service → Indexers")


if __name__ == "__main__":
    run_skillset_pipeline()
