"""
Central configuration for the EDI 814 RAG Chatbot backend.
All secrets are loaded from environment variables — never hardcode keys.
"""

import os
from dataclasses import dataclass


@dataclass
class AzureOpenAIConfig:
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")
    embedding_dimensions: int = 1536


@dataclass
class AzureSearchConfig:
    endpoint: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    api_key: str = os.getenv("AZURE_SEARCH_API_KEY", "")
    api_version: str = "2024-05-01-preview"
    index_name: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "edi814-index")
    indexer_name: str = "edi814-indexer"
    skillset_name: str = "edi814-skillset"
    datasource_name: str = "edi814-datasource"


@dataclass
class AzureBlobConfig:
    connection_string: str = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
    container_name: str = os.getenv("AZURE_BLOB_CONTAINER", "edi814-docs")
    account_name: str = os.getenv("AZURE_BLOB_ACCOUNT_NAME", "")


@dataclass
class AzureDocIntelligenceConfig:
    endpoint: str = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", "")
    api_key: str = os.getenv("AZURE_DOC_INTELLIGENCE_KEY", "")
    api_version: str = "2024-02-29-preview"


@dataclass
class ChunkingConfig:
    # Semantic chunking parameters
    chunk_size: int = 512           # Max tokens per chunk
    chunk_overlap: int = 64         # Token overlap between chunks
    min_chunk_size: int = 100       # Discard chunks smaller than this
    # EDI-specific boundaries — chunk at these logical boundaries
    semantic_boundaries: list = None

    def __post_init__(self):
        self.semantic_boundaries = [
            r"^#{1,3}\s",           # Markdown headers
            r"^Segment:",           # EDI segment definitions
            r"^Element:",           # EDI element definitions
            r"^REF\*",              # Reference segment codes
            r"^DTM\*",              # Date/time segments
            r"^N1\*",               # Name segments
            r"^<!--\s*Page\s*\d+",  # Page breaks
            r"Version \d+\.\d+",    # Version markers
        ]


# Singleton instances
openai_cfg = AzureOpenAIConfig()
search_cfg = AzureSearchConfig()
blob_cfg = AzureBlobConfig()
doc_intel_cfg = AzureDocIntelligenceConfig()
chunking_cfg = ChunkingConfig()
