"""
chatbot_app.py  —  EDI 814 Drop Transaction Chatbot
----------------------------------------------------
Streamlit UI that wraps Azure AI Search (hybrid + semantic reranking)
+ GPT-4 RAG to answer questions over your indexed EDI/utility documents.

Environment variables (put in .env or export before running):
  AZURE_SEARCH_ENDPOINT
  AZURE_SEARCH_ADMIN_KEY
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_EMBED_DEPLOY     (default: text-embedding-3-small)
  AZURE_OPENAI_CHAT_DEPLOY      (default: gpt-4o-mini)
  SEARCH_INDEX_NAME             (default: edi-documents)

Run:
  streamlit run chatbot_app.py
"""

import os
import re
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EDI Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.stApp { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f3e; }
[data-testid="stChatMessage"] {
    background: #1a1f2e; border-radius: 12px;
    margin-bottom: 8px; padding: 4px 8px; border: 1px solid #2a2f3e;
}
.source-card {
    background: #1e2436; border: 1px solid #2e3650;
    border-left: 3px solid #4f8ef7; border-radius: 8px;
    padding: 10px 14px; margin: 6px 0; font-size: 0.85rem;
}
.source-card .section-path { color: #8892b0; font-size: 0.78rem; margin-bottom: 4px; }
.source-card .doc-link { color: #4f8ef7; text-decoration: none; font-weight: 500; }
.source-card .doc-link:hover { text-decoration: underline; }
.badge {
    display: inline-block; padding: 1px 8px; border-radius: 12px;
    font-size: 0.72rem; font-weight: 600; margin-left: 6px;
}
.badge-table  { background: #1a3a5c; color: #60b4ff; }
.badge-figure { background: #2a1f4a; color: #b47aff; }
.badge-text   { background: #1a3a2a; color: #60d48a; }
.badge-list   { background: #3a2a1a; color: #ffaa60; }
.enrich-badge {
    display: inline-block; background: #1f2d1f; color: #6fcf97;
    border: 1px solid #2d5a2d; border-radius: 6px;
    padding: 2px 8px; font-size: 0.72rem; margin-top: 4px;
}
.filter-badge {
    display: inline-block; background: #1f1f2d; color: #9b8dff;
    border: 1px solid #3a2d5a; border-radius: 6px;
    padding: 2px 8px; font-size: 0.72rem; margin-top: 4px;
}
.metric-card {
    background: #1a1f2e; border: 1px solid #2a2f3e;
    border-radius: 10px; padding: 12px 16px; text-align: center;
}
.metric-val   { font-size: 1.6rem; font-weight: 700; color: #4f8ef7; }
.metric-label { font-size: 0.75rem; color: #8892b0; margin-top: 2px; }
.stChatInput > div { background: #1a1f2e !important; border-color: #2a2f3e !important; }
h1, h2, h3 { color: #e2e8f0 !important; }
p, li, div { color: #c8cdd8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Azure client initialisation (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to Azure…")
def get_clients():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from openai import AzureOpenAI

    endpoint   = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
    key        = os.environ.get("AZURE_SEARCH_ADMIN_KEY", "")
    index_name = os.environ.get("SEARCH_INDEX_NAME", "edi-documents")
    aoai_ep    = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    aoai_key   = os.environ.get("AZURE_OPENAI_API_KEY", "")

    if not all([endpoint, key, aoai_ep, aoai_key]):
        return None, None, None, None

    cred          = AzureKeyCredential(key)
    search_client = SearchClient(endpoint, index_name, cred)
    openai_client = AzureOpenAI(
        azure_endpoint=aoai_ep,
        api_key=aoai_key,
        api_version="2024-02-01",
    )
    embed_deploy = os.environ.get("AZURE_OPENAI_EMBED_DEPLOY", "text-embedding-3-small")
    chat_deploy  = os.environ.get("AZURE_OPENAI_CHAT_DEPLOY", "gpt-4o-mini")
    return search_client, openai_client, embed_deploy, chat_deploy


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — Metadata Pre-filtering
#
# WHY: Without filters, BM25 searches ALL documents. A question about
# "Central Hudson nomination deadlines" can return high-scoring chunks
# from Con Edison or National Grid that mention "deadlines" — crowding
# out the correct document entirely.
#
# HOW: Scan the question for utility and topic keywords, build an OData
# filter, and pass it to Azure Search BEFORE retrieval begins. This
# narrows the candidate pool to only the relevant utility/topic.
#
# CONFIGURE: Update UTILITY_KEYWORD_MAP values to match exactly what is
# stored in your index's source_pdf_name field.
# ─────────────────────────────────────────────────────────────────────────────

UTILITY_KEYWORD_MAP: dict[str, str] = {
    # keyword (lowercase)          → source_pdf_name value in your index
    "central hudson":              "Central Hudson Gas & Electric",
    "central hud":                 "Central Hudson Gas & Electric",
    "con edison":                  "Con Edison",
    "coned":                       "Con Edison",
    "national fuel":               "National Fuel Gas",
    "national grid":               "National Grid",
    "nyseg":                       "New York State Electric & Gas",
    "new york state electric":     "New York State Electric & Gas",
    "orange and rockland":         "Orange and Rockland Utilities",
    "o&r":                         "Orange and Rockland Utilities",
    "pseg long island":            "PSEG Long Island",
    "pseg":                        "PSEG Long Island",
    "rochester gas":               "Rochester Gas and Electric",
    "rge":                         "Rochester Gas and Electric",
}

TOPIC_KEYWORD_MAP: dict[str, str] = {
    # keyword (lowercase)    → topic field value in your index
    "nomination":            "Nominations",
    "balancing":             "Balancing",
    "imbalance":             "Balancing",
    "capacity release":      "Capacity Release",
    "drop transaction":      "Drop Transaction",
    "edi 814":               "Drop Transaction",
    "814":                   "Drop Transaction",
    "enrollment":            "Enrollment",
    "enroll":                "Enrollment",
    "billing":               "Billing",
    "invoice":               "Billing",
    "transportation":        "Transportation",
    "supplier":              "Supplier Requirements",
    "esco":                  "Supplier Requirements",
    "retail access":         "Retail Access",
    "switching":             "Switching",
    "renewal":               "Contract Renewal",
    "record retention":      "Record Retention",
    "retention":             "Record Retention",
    "cramming":              "Consumer Protection",
    "disconnection":         "Disconnection",
    "low income":            "Low Income Programs",
    "por":                   "POR",
    "non-por":               "Non-POR",
}


def detect_metadata_from_question(question: str) -> dict[str, Optional[str]]:
    """
    Scan the question text for known utility and topic keywords.
    Returns {'utility': str|None, 'topic': str|None}.
    Longer keywords are matched first so "pseg long island" beats "pseg".
    """
    q = question.lower()

    detected_utility: Optional[str] = None
    for kw in sorted(UTILITY_KEYWORD_MAP, key=len, reverse=True):
        if kw in q:
            detected_utility = UTILITY_KEYWORD_MAP[kw]
            break

    detected_topic: Optional[str] = None
    for kw in sorted(TOPIC_KEYWORD_MAP, key=len, reverse=True):
        if kw in q:
            detected_topic = TOPIC_KEYWORD_MAP[kw]
            break

    return {"utility": detected_utility, "topic": detected_topic}


def build_odata_filter(
    detected: dict[str, Optional[str]],
    filter_content_type: Optional[str],
    filter_source: Optional[str],          # manual sidebar override
) -> Optional[str]:
    """
    Merge auto-detected filters + manual sidebar filters into one OData string.
    Manual filter_source always overrides auto-detected utility.
    """
    filters: list[str] = []

    if filter_content_type and filter_content_type != "All":
        filters.append(f"content_type eq '{filter_content_type.lower()}'")

    # Manual sidebar document filter takes priority
    if filter_source and filter_source.strip():
        safe = filter_source.replace("'", "''")
        filters.append(f"source_pdf_name eq '{safe}'")
    elif detected.get("utility"):
        safe = detected["utility"].replace("'", "''")
        filters.append(f"source_pdf_name eq '{safe}'")

    # Uncomment below if your index has a filterable 'topic' field:
    # if detected.get("topic"):
    #     safe_topic = detected["topic"].replace("'", "''")
    #     filters.append(f"topic eq '{safe_topic}'")

    return " and ".join(filters) if filters else None


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — Contextual Vector Enrichment (embedding only, BM25 untouched)
#
# WHY: Your question "What are the nomination deadlines?" is a short sentence.
# Its embedding sits near other short questions in vector space — far from
# your indexed chunks, which are long paragraphs tagged with rich metadata
# like "Section 4.2 > Nominations > Daily Scheduling > ...".
#
# HOW: Prepend the detected utility and topic as a breadcrumb to the question
# BEFORE embedding it. This pulls the query vector into the same region of
# the embedding space as your indexed chunks.
#
#   Raw:      "What are the nomination deadlines?"
#   Enriched: "Central Hudson Gas & Electric | Nominations |
#              What are the nomination deadlines?"
#
# The raw question is still sent to BM25 unchanged — only the vector query
# is enriched. No LLM call needed.
# ─────────────────────────────────────────────────────────────────────────────

def build_enriched_query(question: str, detected: dict[str, Optional[str]]) -> str:
    prefix_parts = [p for p in [detected.get("utility"), detected.get("topic")] if p]
    if not prefix_parts:
        return question
    return " | ".join(prefix_parts) + " | " + question


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────

def embed_query(text: str, client, deploy: str) -> list[float]:
    resp = client.embeddings.create(input=[text], model=deploy)
    return resp.data[0].embedding


def hybrid_search(
    query: str,
    search_client,
    openai_client,
    embed_deploy: str,
    top_k: int = 6,
    filter_content_type: Optional[str] = None,
    filter_source: Optional[str] = None,
    detected_metadata: Optional[dict] = None,
    score_threshold: float = 0,
    use_metadata_filter: bool = True,
    use_contextual_enrichment: bool = True,
) -> tuple[list[dict], dict]:
    """
    Returns (hits, debug_info).
    """
    from azure.search.documents.models import VectorizedQuery

    detected = detected_metadata or {"utility": None, "topic": None}

    # FIX 2: build enriched query for embedding
    if use_contextual_enrichment:
        enriched_query = build_enriched_query(query, detected)
    else:
        enriched_query = query

    # FIX 1: build OData filter
    odata_filter = build_odata_filter(
        detected if use_metadata_filter else {"utility": None, "topic": None},
        filter_content_type,
        filter_source,
    )

    vector = embed_query(enriched_query, openai_client, embed_deploy)
    candidate_k = 50 # max(top_k * 3, 15)

    vq = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=candidate_k,
        fields="content_vector",
    )

    results = search_client.search(
        search_text=query,   # raw question for BM25 — intentionally unchanged
        vector_queries=[vq],
        search_fields=["content", "section_title", "section", "topic", "subtopic"],
        query_type="semantic",
        semantic_configuration_name="semantic-config",
        query_caption="extractive",
        query_answer="extractive",
        top=candidate_k,
        filter=odata_filter,
        select=[
            "chunk_id", "source", "source_pdf_url", "source_pdf_name",
            "chunk_index", "page_start", "page_end",
            "section_title", "section", "topic", "subtopic",
            "content", "content_type", "is_table", "is_figure",
            "metadata_json",
        ],
    )

    hits: list[dict] = []
    seen: set[str] = set()

    for r in results:
        d = dict(r)
        reranker = r.get("@search.reranker_score")
        st.write(f"Debug: BM25 score={r.get('@search.score', 0):.2f}, Reranker score={reranker:.2f}" if reranker is not None else f"Debug: BM25 score={r.get('@search.score', 0):.2f}, Reranker score=None")
        bm25     = r.get("@search.score", 0)
        d["_score"]          = reranker if reranker is not None else bm25
        d["_reranker_score"] = reranker
        d["_bm25_score"]     = bm25

        if reranker is not None and score_threshold > 0 and reranker < score_threshold:
            continue

        fp = d.get("content", "")[:200].strip()
        if fp in seen:
            continue
        seen.add(fp)

        captions = r.get("@search.captions", [])
        if captions:
            d["_caption"] = captions[0].text

        hits.append(d)

    hits.sort(key=lambda x: x["_score"], reverse=True)

    debug = {
        "enriched_query": enriched_query,
        "odata_filter":   odata_filter,
        "detected":       detected,
    }
    return hits[:top_k], debug


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4 RAG
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a Retail Energy Regulatory and Market Rules Assistant.

Your role is to provide accurate, market-specific, and utility-specific regulatory guidance for retail energy operations in U.S. deregulated electricity and natural gas markets.

Your answers must prioritize regulatory accuracy over completeness.

---------------------------------------------------------------------

REGULATORY DISCLAIMER (FIRST RESPONSE ONLY)

In the first response of every new conversation, include the following disclaimer exactly once:

Regulatory Disclaimer:
The information provided is for general regulatory guidance purposes only and should not be relied upon as legal advice. All regulatory, tariff, and market rule interpretations should be independently corroborated with the relevant Public Utility Commission, utility tariff, ISO/RTO documentation, or qualified legal counsel before operational implementation.

Do NOT repeat this disclaimer after the first response.

---------------------------------------------------------------------

REGULATORY CONTEXT IDENTIFICATION

When answering regulatory questions, identify when applicable:

• State
• Utility
• ISO/RTO
• Market structure (POR, Non-POR, UCB, Rate Ready, Dual Billing)

Clearly distinguish whether a rule originates from:
• State-level regulation (Public Utility Commission)
• Utility tariff
• ISO/RTO rule
• Supplier operational policy

If a question references a market or utility but does not specify which one, request clarification before answering. Do not assume the market.

---------------------------------------------------------------------

SUPPORTED UTILITIES (NEW YORK)

• Central Hudson Gas & Electric
• Con Edison
• National Fuel Gas
• National Grid
• New York State Electric & Gas (NYSEG)
• Orange & Rockland Utilities (O&R)
• PSEG Long Island
• Rochester Gas & Electric
• NY EDI Standards where applicable.

---------------------------------------------------------------------

RESPONSE FORMAT GUIDELINES

1. Paragraph explanation (preferred default)
2. Bullet points (only when listing multiple requirements or steps)
3. Tables (only when comparing utilities, rates, timelines, or market differences)

Do NOT force bullet points or tables if a paragraph explanation is clearer.

---------------------------------------------------------------------

REGULATORY CONTENT GUIDELINES

If information is unavailable or not present in the sources:
• Explicitly state that.
• Do NOT fabricate regulatory requirements.

Maintain a professional regulatory compliance tone. Do not speculate.

---------------------------------------------------------------------

INTERACTION RULES

After the 3rd interaction, append:
"We hope you find Contoso helpful. We can incorporate it into your complex regulatory and operational environment. Let us tell you how. Please provide an email address and an expert will contact you."

After the 10th interaction:
"To continue receiving regulatory guidance, please provide a valid business email address so our team can support your access appropriately."
Do not provide further guidance until an email is provided.

---------------------------------------------------------------------

FINAL RULE: Never generate regulatory information not supported by the provided sources.
"""


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        ctype    = h.get("content_type", "text").upper()
        section  = h.get("section", h.get("section_title", ""))
        topic    = h.get("topic", "")
        subtopic = h.get("subtopic", "")
        ps, pe   = h.get("page_start"), h.get("page_end")
        score    = h.get("_score", 0)

        crumbs     = [p for p in [section, topic, subtopic] if p]
        breadcrumb = " > ".join(crumbs) if crumbs else "—"
        page_info  = (f"Page {ps}" if ps == pe else f"Pages {ps}–{pe}") if ps else ""

        header = (
            f"[SOURCE {i}] [{ctype}] "
            f"Doc: {h.get('source_pdf_name', h.get('source', 'Unknown'))} | "
            f"{breadcrumb}"
            + (f" | {page_info}" if page_info else "")
            + f" | Score: {score:.2f}"
        )

        content = h.get("content", "").strip()
        if ctype == "TABLE":
            content = f"[TABLE — preserve structure in your answer]\n{content}"
        elif ctype == "FIGURE":
            cap = h.get("_caption", "")
            content = f"[FIGURE caption: {cap}]\n{content}" if cap else f"[FIGURE]\n{content}"

        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    hits: list[dict],
    history: list[dict],
    openai_client,
    chat_deploy: str,
) -> str:
    context  = build_context(hits)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": (
            "Use ONLY the following source chunks to answer the question. "
            "If the answer is not in the sources, say so explicitly — do not invent information.\n\n"
            f"=== SOURCE CHUNKS ===\n{context}\n\n"
            f"=== QUESTION ===\n{question}"
        ),
    })

    response = openai_client.chat.completions.create(
        model=chat_deploy,
        messages=messages,
        temperature=0.1,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

BADGE_MAP = {
    "table":  ("TABLE",  "badge-table"),
    "figure": ("FIGURE", "badge-figure"),
    "list":   ("LIST",   "badge-list"),
    "mixed":  ("MIXED",  "badge-text"),
    "text":   ("TEXT",   "badge-text"),
}


def render_source_card(hit: dict, index: int):
    ctype      = hit.get("content_type", "text").lower()
    label, cls = BADGE_MAP.get(ctype, ("TEXT", "badge-text"))
    section    = hit.get("section", hit.get("section_title", "—"))
    topic      = hit.get("topic", "")
    subtopic   = hit.get("subtopic", "")
    ps, pe     = hit.get("page_start"), hit.get("page_end")
    pdf_url    = hit.get("source_pdf_url", "")
    pdf_name   = hit.get("source_pdf_name", hit.get("source", "Document"))
    score      = hit.get("_score", 0)
    reranker   = hit.get("_reranker_score")
    caption    = hit.get("_caption", "")

    page_str   = (f"p.{ps}" if ps == pe else f"pp.{ps}–{pe}") if ps else ""
    crumbs     = [p for p in [section, topic, subtopic] if p]
    breadcrumb = " › ".join(crumbs) if crumbs else "—"
    link_html  = (
        f'<a class="doc-link" href="{pdf_url}" target="_blank">{pdf_name}</a>'
        if pdf_url else f'<span class="doc-link">{pdf_name}</span>'
    )
    score_str    = f"{score:.2f}" if score else "—"
    reranker_str = f" | reranker: {reranker:.2f}" if reranker is not None else ""

    st.markdown(f"""
    <div class="source-card">
        <div class="section-path">
            {link_html}
            <span class="badge {cls}">{label}</span>
            {"&nbsp;·&nbsp;" + page_str if page_str else ""}
            &nbsp;·&nbsp; score {score_str}{reranker_str}
        </div>
        <div style="color:#a0aec0; font-size:0.82rem; margin-bottom:4px;">📂 {breadcrumb}</div>
        {f'<div style="color:#cbd5e0; font-size:0.83rem; font-style:italic;">💬 {caption}</div>' if caption else ""}
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"View chunk {index} content"):
        st.markdown(hit.get("content", ""), unsafe_allow_html=False)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_sources" not in st.session_state:
    st.session_state.total_sources = 0


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ EDI Chatbot")
    st.markdown("---")

    st.markdown("### 🔌 Connection")
    search_client, openai_client, embed_deploy, chat_deploy = get_clients()

    if search_client:
        st.success("Azure Search ✓")
        st.success("Azure OpenAI ✓")
    else:
        st.error("Missing environment variables")
        with st.expander("Required variables"):
            st.code("""AZURE_SEARCH_ENDPOINT
AZURE_SEARCH_ADMIN_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_EMBED_DEPLOY
AZURE_OPENAI_CHAT_DEPLOY
SEARCH_INDEX_NAME""")

    st.markdown("---")
    st.markdown("### 🔧 Search Settings")

    top_k = st.slider("Results to retrieve", 3, 10, 6)

    score_threshold = st.slider(
        "Min reranker score (0 = off)",
        min_value=0.0, max_value=4.0, value=1.0, step=0.1,
        help="Chunks below this reranker score are dropped. Set to 0 to disable.",
    )

    use_metadata_filter = st.toggle(
        "Auto metadata filtering",
        value=True,
        help=(
            "Detects utility name from your question and pre-filters the index. "
            "Eliminates wrong-document chunks. Disable when asking about multiple utilities."
        ),
    )

    use_contextual_enrichment = st.toggle(
        "Contextual vector enrichment",
        value=True,
        help=(
            "Prepends detected utility + topic as a breadcrumb to the question "
            "before embedding. Pulls the vector into the right chunk neighbourhood. "
            "Only affects the vector query — BM25 stays unchanged."
        ),
    )

    filter_type = st.selectbox(
        "Filter by content type",
        ["All", "Text", "Table", "Figure", "List"],
    )

    filter_doc = st.text_input(
        "Filter by document name (overrides auto-detect)",
        placeholder="Leave blank for all docs",
    )

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{st.session_state.total_queries}</div>
            <div class="metric-label">Queries</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{st.session_state.total_sources}</div>
            <div class="metric-label">Sources Used</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_sources = 0
        st.rerun()

    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "What is the EDI 814 Drop Transaction process?",
        "How does Central Hudson handle capacity releases?",
        "What are the nomination submission deadlines?",
        "Explain the balancing and imbalance charges.",
        "What documentation is required for a new supplier?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state._pending_question = s
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# ⚡ EDI 814 Drop Transaction Chatbot")
st.markdown(
    "Ask questions about EDI procedures, Central Hudson Gas & Electric "
    "transportation operating procedures, and related utility documents."
)
st.markdown("---")

# ── Render history ─────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "⚡"):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("debug"):
            debug = msg["debug"]
            det   = debug.get("detected", {})
            badges = []
            if det.get("utility"):
                badges.append(f'<span class="filter-badge">🏢 {det["utility"]}</span>')
            if det.get("topic"):
                badges.append(f'<span class="filter-badge">📌 {det["topic"]}</span>')
            eq = debug.get("enriched_query", "")
            if eq:
                short = eq[:70] + ("…" if len(eq) > 70 else "")
                badges.append(f'<span class="enrich-badge">🔍 {short}</span>')
            if badges:
                st.markdown(" &nbsp; ".join(badges), unsafe_allow_html=True)

        if msg["role"] == "assistant" and msg.get("hits"):
            with st.expander(f"📚 View {len(msg['hits'])} source chunks", expanded=False):
                for i, hit in enumerate(msg["hits"], 1):
                    render_source_card(hit, i)


# ── Chat input ─────────────────────────────────────────────────────────────

pending = getattr(st.session_state, "_pending_question", None)
if pending:
    del st.session_state._pending_question
    user_input = pending
else:
    user_input = st.chat_input(
        "Ask about EDI 814 procedures, Central Hudson rules, nominations, balancing…",
        disabled=(search_client is None),
    )

if user_input:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    plain_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant", avatar="⚡"):
        answer_placeholder = st.empty()
        source_placeholder = st.empty()

        # Detect metadata from raw question
        detected = detect_metadata_from_question(user_input)

        # Show detection badges immediately
        if detected.get("utility") or detected.get("topic"):
            badge_html = []
            if detected.get("utility"):
                tag = "🔒 Filtered" if use_metadata_filter else "🔍 Detected"
                badge_html.append(f'<span class="filter-badge">{tag}: {detected["utility"]}</span>')
            if detected.get("topic"):
                badge_html.append(f'<span class="filter-badge">📌 {detected["topic"]}</span>')
            st.markdown(" &nbsp; ".join(badge_html), unsafe_allow_html=True)

        with st.spinner("Searching documents…"):
            try:
                hits, debug_info = hybrid_search(
                    query=user_input,
                    search_client=search_client,
                    openai_client=openai_client,
                    embed_deploy=embed_deploy,
                    top_k=top_k,
                    filter_content_type=filter_type if filter_type != "All" else None,
                    filter_source=filter_doc if filter_doc.strip() else None,
                    detected_metadata=detected,
                    score_threshold=score_threshold,
                    use_metadata_filter=use_metadata_filter,
                    use_contextual_enrichment=use_contextual_enrichment,
                )

                eq = debug_info.get("enriched_query", "")
                if eq and eq != user_input and use_contextual_enrichment:
                    short = eq[:80] + ("…" if len(eq) > 80 else "")
                    st.markdown(
                        f'<span class="enrich-badge">🔍 Embedded as: {short}</span>',
                        unsafe_allow_html=True,
                    )
                if debug_info.get("odata_filter"):
                    st.caption(f"🔒 OData filter: `{debug_info['odata_filter']}`")

            except Exception as e:
                st.error(f"Search error: {e}")
                hits, debug_info = [], {}

        if not hits:
            answer = (
                "I couldn't find relevant information for that question in the indexed documents. "
                "If **Auto metadata filtering** is on and your question spans multiple utilities, "
                "try disabling it in the sidebar. You can also lower the min reranker score."
            )
        else:
            with st.spinner("Generating answer…"):
                try:
                    answer = generate_answer(
                        question=user_input,
                        hits=hits,
                        history=plain_history,
                        openai_client=openai_client,
                        chat_deploy=chat_deploy,
                    )
                except Exception as e:
                    answer = f"Generation error: {e}"

        answer_placeholder.markdown(answer)

        if hits:
            with source_placeholder.expander(f"📚 {len(hits)} source chunks used", expanded=True):
                for i, hit in enumerate(hits, 1):
                    render_source_card(hit, i)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "hits": hits,
        "debug": {**debug_info, "detected": detected},
    })
    st.session_state.total_queries += 1
    st.session_state.total_sources += len(hits)