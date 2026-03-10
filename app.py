"""
app.py
------
FastAPI RAG chatbot backend.

Endpoints:
  POST /chat          → single-turn Q&A
  POST /chat/stream   → streaming response
  GET  /health        → health check
  GET  /sources       → list indexed source files

Run:
  uvicorn app:app --reload --port 8000
"""

import os
import json
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import AzureOpenAI

from retriever import retrieve, format_context, RetrievedChunk

from dotenv import load_dotenv
load_dotenv()
# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "EDI Document Chatbot",
    description = "RAG chatbot over NY EDI 814 standards documents",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # tighten in production
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── LLM client ────────────────────────────────────────────────────────────────

def _get_llm():
    return AzureOpenAI(
        azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key        = os.environ["AZURE_OPENAI_API_KEY"],
        api_version    = "2024-02-01",
    )

CHAT_DEPLOY = os.environ.get("AZURE_OPENAI_CHAT_DEPLOY", "gpt-4o")


# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert assistant for NY State Energy Deregulation EDI standards.
Answer questions based ONLY on the provided context from official standards documents.

Rules:
- Be precise and cite the source document and section when possible.
- If the answer is not in the context, say "I don't have enough information in the indexed documents to answer this."
- For table data (segment codes, element values), present them clearly.
- Keep answers concise but complete.
- If a question involves multiple documents or versions, compare them clearly.
"""


def build_prompt(query: str, context: str, history: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include conversation history (last 6 turns to stay within context window)
    messages.extend(history[-6:])

    # Add context + current question
    user_message = f"""Context from indexed documents:
---
{context}
---

Question: {query}"""

    messages.append({"role": "user", "content": user_message})
    return messages


# ── Request / Response models ─────────────────────────────────────────────────

class Message(BaseModel):
    role:    str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query:              str
    history:            list[Message] = Field(default_factory=list)
    top_k:              int           = Field(default=5, ge=1, le=20)
    filter_source:      Optional[str] = None
    filter_tables_only: bool          = False
    stream:             bool          = False


class SourceReference(BaseModel):
    chunk_id: str
    source:   str
    section:  str
    page:     int
    score:    float


class ChatResponse(BaseModel):
    answer:   str
    sources:  list[SourceReference]
    query:    str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "EDI RAG Chatbot"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Single-turn RAG Q&A."""
    # 1. Retrieve relevant chunks
    chunks = retrieve(
        query              = request.query,
        top_k              = request.top_k,
        filter_source      = request.filter_source,
        filter_tables_only = request.filter_tables_only,
    )

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    # 2. Build context
    context = format_context(chunks)

    # 3. Build messages
    history  = [m.model_dump() for m in request.history]
    messages = build_prompt(request.query, context, history)

    # 4. Call LLM
    llm      = _get_llm()
    response = llm.chat.completions.create(
        model       = CHAT_DEPLOY,
        messages    = messages,
        temperature = 0.1,
        max_tokens  = 1000,
    )

    answer = response.choices[0].message.content

    sources = [
        SourceReference(
            chunk_id = c.chunk_id,
            source   = c.source,
            section  = c.section,
            page     = c.page,
            score    = round(c.score, 4),
        )
        for c in chunks
    ]

    return ChatResponse(answer=answer, sources=sources, query=request.query)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming RAG Q&A — returns server-sent events."""

    chunks = retrieve(
        query              = request.query,
        top_k              = request.top_k,
        filter_source      = request.filter_source,
        filter_tables_only = request.filter_tables_only,
    )

    context  = format_context(chunks) if chunks else "No context found."
    history  = [m.model_dump() for m in request.history]
    messages = build_prompt(request.query, context, history)

    llm = _get_llm()

    async def event_generator() -> AsyncGenerator[str, None]:
        # First, send source metadata
        sources = [
            {
                "chunk_id": c.chunk_id,
                "source":   c.source,
                "section":  c.section,
                "page":     c.page,
                "score":    round(c.score, 4),
            }
            for c in chunks
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Then stream the answer
        stream = llm.chat.completions.create(
            model       = CHAT_DEPLOY,
            messages    = messages,
            temperature = 0.1,
            max_tokens  = 1000,
            stream      = True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield f"data: {json.dumps({'type': 'token', 'text': delta.content})}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/sources")
def list_sources():
    """List all indexed source files."""
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    key      = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    client   = SearchClient(endpoint, "edi-documents", AzureKeyCredential(key))

    results  = client.search(
        search_text = "*",
        select      = ["source"],
        top         = 1000,
    )
    sources = sorted(set(r["source"] for r in results if r.get("source")))
    return {"sources": sources, "count": len(sources)}
