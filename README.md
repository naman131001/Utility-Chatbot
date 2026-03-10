# EDI Document RAG Chatbot — Setup Guide

## Architecture

```
Markdown Files (Azure DI output)
        ↓
[chunker.py]  Semantic chunking (heading-aware, table-preserving)
        ↓
[indexer.py]  Embed → Azure AI Search (vector + keyword index)
        ↓
[retriever.py] Hybrid search (vector + BM25) on every query
        ↓
[app.py]      FastAPI: /chat endpoint → GPT-4o generates answer
```

---

## Prerequisites

| Service | Tier needed |
|---------|-------------|
| Azure AI Search | Basic+ (Free tier has no vector search) |
| Azure OpenAI | Any tier with `text-embedding-3-small` + `gpt-4o` deployments |
| Python | 3.11+ |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your keys in .env
```

Load it before running:
```bash
export $(cat .env | xargs)
# or use python-dotenv (already imported in app.py if you add load_dotenv())
```

### 3. Index your documents

Put all your markdown files (from Azure Document Intelligence) in a folder:
```
documents/
  814D_Drop_Transaction_v16.md
  814E_Enroll_Transaction_v12.md
  ...
```

Run the indexer:
```bash
python indexer.py ./documents
```

This will:
- Create the Azure AI Search index schema (with vector field)
- Chunk every markdown file semantically
- Embed each chunk via Azure OpenAI
- Upload everything to the index

Expected output:
```
Found 12 markdown files in './documents'
  814D_Drop_Transaction_v16.md: 87 chunks
  ...
Total chunks: 943
✅ Index 'edi-documents' created/updated.
Uploading 943 chunks ...
  Batch 1: 100/100 uploaded
  ...
✅ Indexing complete.
```

### 4. Start the API server

```bash
uvicorn app:app --reload --port 8000
```

---

## API Usage

### Single-turn Q&A

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the drop reason code for a customer returning to utility full service?",
    "top_k": 5
  }'
```

Response:
```json
{
  "answer": "The drop reason code for a customer returning to utility full service is CHU...",
  "sources": [
    {
      "chunk_id": "814D_Drop_Transaction_v16_0012_abc123",
      "source": "814D_Drop_Transaction_v16.md",
      "section": "REF Reference Identification",
      "page": 4,
      "score": 0.9823
    }
  ],
  "query": "What is the drop reason code..."
}
```

### Multi-turn conversation

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What changed in version 1.6?",
    "history": [
      {"role": "user", "content": "What is the NPD code?"},
      {"role": "assistant", "content": "NPD means No Pending Drop..."}
    ]
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the DTM*151 segment"}'
```

Events:
```
data: {"type": "sources", "sources": [...]}
data: {"type": "token", "text": "The "}
data: {"type": "token", "text": "DTM*151 "}
...
data: {"type": "done"}
```

### Filter by source file

```bash
curl -X POST http://localhost:8000/chat \
  -d '{"query": "...", "filter_source": "814D_Drop_Transaction_v16.md"}'
```

---

## File Reference

| File | Purpose |
|------|---------|
| `chunker.py` | Parse markdown → semantic Chunk objects |
| `indexer.py` | Create Azure AI Search index + upload chunks |
| `skillset.py` | *Alternative*: Azure-native pipeline (Blob → Skillset → Index) |
| `retriever.py` | Hybrid vector+keyword search |
| `app.py` | FastAPI chatbot backend |

---

## Choosing: indexer.py vs skillset.py

| | `indexer.py` (Python pipeline) | `skillset.py` (Azure native) |
|--|--|--|
| Control | Full control over chunking logic | Azure controls chunking |
| Table handling | ✅ Custom table-preserving logic | ❌ Tables may get split |
| New file pickup | Re-run script | Automatic (scheduled) |
| Recommended for | Your use case (DI markdown output) | Simple text blobs |

**Recommendation:** Use `indexer.py` — your DI output has tables and metadata that benefit from custom chunking.

---

## Tuning Tips

- **Chunk size**: In `chunker.py`, adjust `CHUNK_MAX_TOKENS` (default 512). Smaller = more precise retrieval. Larger = more context per chunk.
- **top_k**: Default 5. Increase for complex multi-part questions.
- **Temperature**: Set to 0.0 for maximum factual accuracy on standards documents.
- **Semantic reranking**: If you upgrade to Azure AI Search S1+, set `AZURE_SEARCH_SEMANTIC=true` for better ranking.
