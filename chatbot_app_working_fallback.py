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
  AZURE_OPENAI_CHAT_DEPLOY      (default: gpt-4)
  SEARCH_INDEX_NAME             (default: edi-documents)

Run:
  streamlit run chatbot_app.py
"""

import json
import os
import time
import re
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Lazy imports so missing packages show friendly errors ────────────────────
def _require(pkg: str):
    try:
        return __import__(pkg)
    except ImportError:
        st.error(f"Missing package: `pip install {pkg}`")
        st.stop()

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
/* ── Overall background ── */
.stApp { background: #0f1117; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a2f3e;
}

/* ── Chat message bubbles ── */
[data-testid="stChatMessage"] {
    background: #1a1f2e;
    border-radius: 12px;
    margin-bottom: 8px;
    padding: 4px 8px;
    border: 1px solid #2a2f3e;
}

/* ── Source card ── */
.source-card {
    background: #1e2436;
    border: 1px solid #2e3650;
    border-left: 3px solid #4f8ef7;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
}
.source-card .section-path {
    color: #8892b0;
    font-size: 0.78rem;
    margin-bottom: 4px;
}
.source-card .doc-link {
    color: #4f8ef7;
    text-decoration: none;
    font-weight: 500;
}
.source-card .doc-link:hover { text-decoration: underline; }
.badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 6px;
}
.badge-table  { background: #1a3a5c; color: #60b4ff; }
.badge-figure { background: #2a1f4a; color: #b47aff; }
.badge-text   { background: #1a3a2a; color: #60d48a; }
.badge-list   { background: #3a2a1a; color: #ffaa60; }

/* ── Fallback badge ── */
.fallback-badge {
    display: inline-block;
    background: #2d1f1f;
    color: #ff9f6b;
    border: 1px solid #5a2d2d;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    margin-bottom: 6px;
}

/* ── Metric cards ── */
.metric-card {
    background: #1a1f2e;
    border: 1px solid #2a2f3e;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}
.metric-val  { font-size: 1.6rem; font-weight: 700; color: #4f8ef7; }
.metric-label { font-size: 0.75rem; color: #8892b0; margin-top: 2px; }

/* ── Input area ── */
.stChatInput > div { background: #1a1f2e !important; border-color: #2a2f3e !important; }

/* ── Headings ── */
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
    chat_deploy  = "gpt-4o-mini"
    return search_client, openai_client, embed_deploy, chat_deploy


# ─────────────────────────────────────────────────────────────────────────────
# Utility keyword detection + fallback filter
#
# When the user names a specific utility → filter on source_pdf_name.
# When NO utility is mentioned and no manual filter_doc is set →
#   fall back to the NY ESCO Doc, identified in the index by
#   energy_utility_name = "document_intelligence".
#
# To change the fallback document, update the two constants below.
# ─────────────────────────────────────────────────────────────────────────────

# Keyword → exact source_pdf_name value stored in your index.
# Longer keywords are matched first to avoid short-keyword false positives.
UTILITY_KEYWORD_MAP: dict[str, str] = {
    "central hudson":          "Central Hudson Gas & Electric",
    "central hud":             "Central Hudson Gas & Electric",
    "con edison":              "Con Edison",
    "coned":                   "Con Edison",
    "national fuel":           "National Fuel Gas",
    "national grid":           "National Grid",
    "nyseg":                   "New York State Electric & Gas",
    "new york state electric": "New York State Electric & Gas",
    "orange and rockland":     "Orange and Rockland Utilities",
    "o&r":                     "Orange and Rockland Utilities",
    "pseg long island":        "PSEG Long Island",
    "pseg":                    "PSEG Long Island",
    "rochester gas":           "Rochester Gas and Electric",
    "rge":                     "Rochester Gas and Electric",
}

# Fallback: when no utility keyword is found, restrict to the document
# that carries this field value in the index.
FALLBACK_FILTER_FIELD = "energy_utility_name"
FALLBACK_FILTER_VALUE = "document_intelligence"


def detect_utility_from_question(question: str) -> Optional[str]:
    """
    Returns the source_pdf_name value for the first utility keyword found
    in the question, or None if no utility is mentioned.
    """
    q = question.lower()
    for kw in sorted(UTILITY_KEYWORD_MAP, key=len, reverse=True):
        if kw in q:
            return UTILITY_KEYWORD_MAP[kw]
    return None


def build_odata_filter(
    detected_utility: Optional[str],
    filter_content_type: Optional[str],
    filter_source: Optional[str],   # manual sidebar override
) -> tuple[Optional[str], str]:
    """
    Build the OData filter string and a short label for the UI badge.

    Priority:
      1. Manual filter_doc (sidebar text box)  → source_pdf_name eq '...'
      2. Utility keyword detected in question  → source_pdf_name eq '...'
      3. No utility found                      → energy_utility_name eq 'document_intelligence'

    Returns (odata_filter | None, label_for_badge).
    """
    filters: list[str] = []
    label = "No filter"

    # Content-type filter (independent of utility logic)
    if filter_content_type and filter_content_type != "All":
        filters.append(f"content_type eq '{filter_content_type.lower()}'")

    if filter_source and filter_source.strip():
        # ── Manual override always wins ────────────────────────────────────
        safe = filter_source.replace("'", "''")
        filters.append(f"source_pdf_name eq '{safe}'")
        label = f"Manual: {filter_source.strip()}"

    elif detected_utility:
        # ── Specific utility detected ──────────────────────────────────────
        safe = detected_utility.replace("'", "''")
        filters.append(f"source_pdf_name eq '{safe}'")
        label = f"Utility: {detected_utility}"

    else:
        # ── No utility → fallback to NY ESCO Doc ──────────────────────────
        safe_val = FALLBACK_FILTER_VALUE.replace("'", "''")
        filters.append(f"{FALLBACK_FILTER_FIELD} eq '{safe_val}'")
        label = "Fallback: NY ESCO Doc"

    odata = " and ".join(filters) if filters else None
    return odata, label


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
    top_k: int = 5,
    filter_content_type: Optional[str] = None,
    filter_source: Optional[str] = None,
) -> tuple[list[dict], str]:
    """
    Returns (hits, filter_label).
    filter_label describes which filter was applied (for the UI badge).
    """
    from azure.search.documents.models import VectorizedQuery

    # Detect utility and build filter (includes fallback logic)
    detected_utility = detect_utility_from_question(query)
    odata_filter, filter_label = build_odata_filter(
        detected_utility=detected_utility,
        filter_content_type=filter_content_type,
        filter_source=filter_source,
    )

    vector = embed_query(query, openai_client, embed_deploy)
    vq = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top_k * 2,
        fields="content_vector",
    )

    results = search_client.search(
        search_text=query,
        vector_queries=[vq],
        search_fields=[
            "section_title",
            "topic",
            "subtopic",
            "content",
        ],
        query_type="semantic",
        semantic_configuration_name="semantic-config",
        query_caption="extractive",
        query_answer="extractive",
        top=top_k,
        filter=odata_filter,
        select=[
            "chunk_id", "source", "source_pdf_url", "source_pdf_name",
            "chunk_index", "page_start", "page_end",
            "section_title", "section", "topic", "subtopic",
            "content", "content_type", "is_table", "is_figure",
            "energy_utility_name", "metadata_json",
        ],
    )

    hits = []
    for r in results:
        d = dict(r)
        captions = r.get("@search.captions", [])
        if captions:
            d["_caption"] = captions[0].text
        d["_score"] = r.get("@search.reranker_score") or r.get("@search.score", 0)
        hits.append(d)

    return hits, filter_label


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

When relevant, reference rules from these utilities:

• Central Hudson Gas & Electric
• Con Edison
• National Fuel Gas
• National Grid
• New York State Electric & Gas (NYSEG)
• Orange & Rockland Utilities (O&R)
• PSEG Long Island
• Rochester Gas & Electric

Also reference:
• NY EDI Standards where applicable.

---------------------------------------------------------------------

RESPONSE FORMAT GUIDELINES

Always choose the format that best explains the regulatory rule.

Formatting priority:

1. Paragraph explanation (preferred default)
2. Bullet points (only when listing multiple requirements or steps)
3. Tables (only when comparing utilities, rates, timelines, or market differences)

Formatting rules:

• Start with a short explanatory paragraph whenever possible.
• Use bullet points only when listing multiple conditions, requirements, or procedural steps.
• Use tables only when comparing rules across utilities, markets, timelines, or attributes.
• Do NOT force bullet points or tables if a paragraph explanation is clearer.
• Keep responses concise and focused on the regulatory requirement.
• If rules differ across utilities or markets, clearly explain the differences.

---------------------------------------------------------------------

FORMAT EXAMPLES

Example 1 – Paragraph (Definition)

Question:
What does "cramming" mean?

Answer:
"Cramming" refers to the addition of unauthorized charges to a customer's bill. These charges appear on the bill without the customer's knowledge or consent.

------------------------------------------------

MUST TAKE ANSWER FROM EXAMPLE IF QUESTION IS ABOUT RETENTION REQUIREMENTS. DO NOT MAKE UP ANSWER IF NOT IN SOURCES.

Example 2 – What is the record retention requirement?

Question:
What is the record retention requirement?

Answer:

Under New York PSC Uniform Business Practices (UBP), ESCOs must retain certain customer records for specific periods depending on the type of record.

Customer Consent and Authorization Records:

• Retention period: **2 years or the length of the sales or renewal contract, whichever is longer.**

This requirement applies to records including:

• Express customer consent to price changes
• Customer authorization to enroll
• Documentation of material contract changes
• Third Party Verification (TPV) recordings or voice confirmations
• Written customer agreements

These records must be maintained to demonstrate compliance with PSC consumer protection and enrollment requirements.

------------------------------------------------

Example 3 – Regulatory Timing Requirement

Question:
What is the timing for contract expiration notices?

Answer:

In New York, renewal and expiration notice timing requirements are governed by the **PSC Uniform Business Practices (UBP)** rather than individual utility tariffs.

For **residential and small commercial customers**, the renewal or expiration notice must be sent within the following window:

• Earliest: 60 calendar days before contract expiration  
• Latest: 30 calendar days before contract expiration  

There are no utility-specific renewal notice timing requirements in New York. All utilities defer to PSC Uniform Business Practices (UBP).

There are no provisions directly tied to large C&I customers that are not enrolled under the UBP-governed retail access program.

---------------------------------------------------------------------

REGULATORY CONTENT GUIDELINES

When relevant, include:

• Enrollment timing rules (meter read vs billing cycle)
• POR vs Non-POR differences
• Tax responsibility distinctions
• Disconnection restrictions
• Low-income protections
• Contract notice and renewal requirements

If information is unavailable, unclear, or not present in the source:

• Explicitly state that the information is not available.
• Do NOT fabricate regulatory requirements.

Maintain a professional regulatory compliance tone.
Do not speculate.

---------------------------------------------------------------------

INTERACTION RULES

After the third interaction in a conversation, append the following message at the end of the response:

"We hope you find Contoso helpful. We can incorporate it into your complex regulatory and operational environment. Let us tell you how. Please provide an email address and an expert will contact you."

Do not interrupt regulatory analysis. Append it only at the end.

---------------------------------------------------------------------

AFTER THE TENTH INTERACTION

After the tenth interaction in a conversation:

Inform the user that continued access requires a valid business email address.

Use the following message:

"To continue receiving regulatory guidance, please provide a valid business email address so our team can support your access appropriately."

Do not provide further regulatory guidance until an email address is provided.

You do not need to technically validate the email address.

---------------------------------------------------------------------

FINAL RULE

Never generate regulatory information that is not supported by the provided sources.
"""


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        page_info = ""
        ps, pe = h.get("page_start"), h.get("page_end")
        if ps:
            page_info = f"  (Page {ps})" if ps == pe else f"  (Pages {ps}–{pe})"
        section = h.get("section", h.get("section_title", ""))
        ctype   = h.get("content_type", "text").upper()
        parts.append(
            f"[SOURCE {i}] [{ctype}] {section}{page_info}\n"
            f"{h['content']}\n"
        )
    return "\n---\n".join(parts)


def generate_answer(
    question: str,
    hits: list[dict],
    history: list[dict],
    openai_client,
    chat_deploy: str,
) -> str:
    context = build_context(hits)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (last 6 turns)
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": (
            f"Use the following source chunks to answer the question.\n\n"
            f"=== SOURCE CHUNKS ===\n{context}\n\n"
            f"=== QUESTION ===\n{question}"
        ),
    })

    response = openai_client.chat.completions.create(
        model=chat_deploy,
        messages=messages,
        temperature=0.1,
        max_tokens=1200,
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
    ps, pe     = hit.get("page_start"), hit.get("page_end")
    pdf_url    = hit.get("source_pdf_url", "")
    pdf_name   = hit.get("source_pdf_name", hit.get("source", "Document"))
    score      = hit.get("_score", 0)
    caption    = hit.get("_caption", "")

    page_str = ""
    if ps:
        page_str = f"p.{ps}" if ps == pe else f"pp.{ps}–{pe}"

    link_html = (
        f'<a class="doc-link" href="{pdf_url}" target="_blank">{pdf_name}</a>'
        if pdf_url else
        f'<span class="doc-link">{pdf_name}</span>'
    )

    score_str = f"{score:.2f}" if score else "—"

    st.markdown(f"""
    <div class="source-card">
        <div class="section-path">
            {link_html}
            <span class="badge {cls}">{label}</span>
            {"&nbsp;·&nbsp;" + page_str if page_str else ""}
            &nbsp;·&nbsp; score {score_str}
        </div>
        <div style="color:#a0aec0; font-size:0.82rem; margin-bottom:4px;">
            📂 {section}
        </div>
        {f'<div style="color:#cbd5e0; font-size:0.83rem; font-style:italic;">{caption}</div>' if caption else ""}
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"View chunk {index} content"):
        st.markdown(hit.get("content", ""), unsafe_allow_html=False)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []   # {role, content, hits?}
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

    # ── Connection status ──────────────────────────────────────────────────
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

    # ── Search settings ────────────────────────────────────────────────────
    st.markdown("### 🔧 Search Settings")

    top_k = st.slider("Results to retrieve", 3, 10, 8)

    filter_type = st.selectbox(
        "Filter by content type",
        ["All", "Text", "Table", "Figure", "List"],
    )

    filter_doc = st.text_input(
        "Filter by document name",
        placeholder="Leave blank for auto",
        help=(
            "Override auto-detection. If blank, the chatbot filters to the detected "
            "utility's document, or falls back to NY ESCO Doc when no utility is mentioned."
        ),
    )

    st.markdown("---")

    # ── Stats ──────────────────────────────────────────────────────────────
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{st.session_state.total_queries}</div>
            <div class="metric-label">Queries</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{st.session_state.total_sources}</div>
            <div class="metric-label">Sources Used</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Actions ───────────────────────────────────────────────────────────
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_sources = 0
        st.rerun()

    # ── Suggested questions ───────────────────────────────────────────────
    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "What is the EDI 814 Drop Transaction process?",
        "How does Central Hudson handle capacity releases?",
        "What are the nomination submission deadlines?",
        "Explain the balancing and imbalance charges.",
        "What documentation is required for a new supplier?",
    ]
    for suggestion in suggestions:
        if st.button(suggestion, use_container_width=True, key=f"sug_{suggestion[:20]}"):
            st.session_state._pending_question = suggestion
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

# ── Render conversation history ────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "⚡"):
        st.markdown(msg["content"])

        # Show filter badge that was applied for this response
        if msg["role"] == "assistant" and msg.get("filter_label"):
            fl = msg["filter_label"]
            badge_cls = "fallback-badge" if fl.startswith("Fallback") else "fallback-badge"
            st.markdown(
                f'<span class="fallback-badge">🔒 {fl}</span>',
                unsafe_allow_html=True,
            )

        # Render sources below assistant messages
        if msg["role"] == "assistant" and msg.get("hits"):
            with st.expander(f"📚 View {len(msg['hits'])} source chunks", expanded=False):
                for i, hit in enumerate(msg["hits"], 1):
                    render_source_card(hit, i)


# ── Chat input ─────────────────────────────────────────────────────────────

# Handle suggestion button clicks
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
    # ── Show user message ──────────────────────────────────────────────────
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # ── Search + generate ──────────────────────────────────────────────────
    with st.chat_message("assistant", avatar="⚡"):
        answer_placeholder = st.empty()
        source_placeholder = st.empty()

        with st.spinner("Searching documents…"):
            try:
                hits, filter_label = hybrid_search(
                    query=user_input,
                    search_client=search_client,
                    openai_client=openai_client,
                    embed_deploy=embed_deploy,
                    top_k=top_k,
                    filter_content_type=filter_type if filter_type != "All" else None,
                    filter_source=filter_doc if filter_doc.strip() else None,
                )

                # Show which filter was applied
                st.markdown(
                    f'<span class="fallback-badge">🔒 {filter_label}</span>',
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Search error: {e}")
                hits, filter_label = [], "Error"

        if not hits:
            answer = (
                "I couldn't find relevant information for that question in the indexed documents. "
                "Try rephrasing, or check that the documents have been indexed."
            )
        else:
            with st.spinner("Generating answer…"):
                try:
                    plain_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
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

        # ── Source citations below answer ──────────────────────────────────
        if hits:
            with source_placeholder.expander(
                f"📚 {len(hits)} source chunks used", expanded=True
            ):
                for i, hit in enumerate(hits, 1):
                    render_source_card(hit, i)

    # ── Update session state ───────────────────────────────────────────────
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "hits": hits,
        "filter_label": filter_label,
    })
    st.session_state.total_queries += 1
    st.session_state.total_sources += len(hits)