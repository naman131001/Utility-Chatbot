"""
abog_chatbot_app.py  —  ABOG Board Certification Chatbot
---------------------------------------------------------
Streamlit UI wrapping Azure AI Search (BM25 keyword + vector embedding +
Azure Semantic Reranker) with GPT-4 RAG over indexed ABOG certification docs.

Search pipeline (3 stages):
  Stage 1 — BM25 keyword search  : lexical match on page_title, section,
                                    breadcrumb, section_h2, section_h3, content
  Stage 2 — Vector embedding      : cosine similarity via content_vector field
                                    using text-embedding-3-small
  Stage 3 — Azure Semantic Reranker: re-scores fused BM25+vector results with a
                                    cross-attention transformer model
  (Optional) GPT cross-encoder    : further reranks top-K by the exact question

Index schema fields (from indexed documents):
  id, chunk_index, chunk_total, url, source_file, page_title, page_type,
  section, chunk_type, section_h2, section_h3, breadcrumb, content,
  content_plain, content_with_context, word_count, token_count,
  has_table, is_callout, is_synthetic, contains_dates, year_mentioned,
  prev_chunk_id, next_chunk_id, last_scraped, indexed_at,
  hypothetical_questions, @search.score

Environment variables (.env or export):
  AZURE_SEARCH_ENDPOINT
  AZURE_SEARCH_ADMIN_KEY
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_EMBED_DEPLOY     (default: text-embedding-3-small)
  AZURE_OPENAI_CHAT_DEPLOY      (default: gpt-4o-mini)
  SEARCH_INDEX_NAME             (default: abog-documents)

Run:
  streamlit run abog_chatbot_app.py
"""

import json
import os
import re
import time
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ABOG Certification Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Clean light theme styling
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Root / background ── */
.stApp {
    background: #f7f9fc;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label {
    color: #374151 !important;
}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    margin-bottom: 10px;
    padding: 6px 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ── Source card ── */
.source-card {
    background: #ffffff;
    border: 1px solid #d1dbe8;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 11px 15px;
    margin: 7px 0;
    font-size: 0.85rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.source-card .doc-link {
    color: #2563eb;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.88rem;
}
.source-card .doc-link:hover { text-decoration: underline; }
.source-card .breadcrumb-text {
    color: #6b7280;
    font-size: 0.78rem;
    margin-top: 3px;
}

/* ── Score pills ── */
.score-pill {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 5px;
    margin-top: 4px;
}
.pill-bm25     { background: #dbeafe; color: #1e40af; }
.pill-vector   { background: #d1fae5; color: #065f46; }
.pill-semantic { background: #ede9fe; color: #5b21b6; }
.pill-gpt      { background: #fef3c7; color: #92400e; }

/* ── Content type badges ── */
.badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 0.70rem;
    font-weight: 600;
    margin-left: 5px;
    vertical-align: middle;
}
.badge-table   { background: #dbeafe; color: #1e40af; }
.badge-policy  { background: #ede9fe; color: #5b21b6; }
.badge-faq     { background: #d1fae5; color: #065f46; }
.badge-callout { background: #fef3c7; color: #92400e; }
.badge-text    { background: #f3f4f6; color: #374151; }

/* ── Pipeline stage bar ── */
.pipeline-row {
    display: flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
    margin: 8px 0 14px 0;
    font-size: 0.77rem;
}
.pipeline-stage {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 3px 10px;
    color: #64748b;
    font-weight: 500;
}
.pipeline-stage.active {
    background: #2563eb;
    border-color: #2563eb;
    color: #ffffff;
    font-weight: 600;
}
.pipeline-arrow { color: #94a3b8; font-size: 0.85rem; }

/* ── Refined query banner ── */
.refined-query-banner {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 3px solid #2563eb;
    border-radius: 6px;
    padding: 7px 12px;
    margin-bottom: 10px;
    font-size: 0.81rem;
    color: #1e40af;
}
.refined-query-banner strong { color: #1d4ed8; }

/* ── Metric cards (sidebar) ── */
.metric-card {
    background: #f0f4ff;
    border: 1px solid #c7d2fe;
    border-radius: 10px;
    padding: 12px 10px;
    text-align: center;
}
.metric-val   { font-size: 1.5rem; font-weight: 700; color: #2563eb; }
.metric-label { font-size: 0.73rem; color: #6b7280; margin-top: 2px; }

/* ── Disclaimer ── */
.disclaimer {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 11px 15px;
    margin-bottom: 14px;
    font-size: 0.82rem;
    color: #78350f;
}
.disclaimer strong { color: #b45309; }

/* ── Chat input ── */
.stChatInput > div {
    background: #ffffff !important;
    border-color: #d1d5db !important;
    border-radius: 10px !important;
}

/* ── General text ── */
h1, h2, h3 { color: #111827 !important; }
p, li { color: #374151; }
.stMarkdown p { color: #374151; }
hr { border-color: #e5e7eb; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    background: #ffffff;
}
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
    index_name = os.environ.get("SEARCH_INDEX_NAME", "abog-documents")
    aoai_ep    = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    aoai_key   = os.environ.get("AZURE_OPENAI_API_KEY", "")

    if not all([endpoint, key, aoai_ep, aoai_key]):
        st.warning(
            "⚠️ Azure credentials not found. Set AZURE_SEARCH_ENDPOINT, "
            "AZURE_SEARCH_ADMIN_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY "
            "in your .env file."
        )
        return None, None, None, None

    cred          = AzureKeyCredential(key)
    search_client = SearchClient(endpoint, index_name, cred)
    openai_client = AzureOpenAI(
        azure_endpoint=aoai_ep,
        api_key=aoai_key,
        api_version="2024-02-01",
    )
    embed_deploy = os.environ.get("AZURE_OPENAI_EMBED_DEPLOY", "text-embedding-3-small")
    chat_deploy  = os.environ.get("AZURE_OPENAI_CHAT_DEPLOY",  "gpt-4o-mini")
    return search_client, openai_client, embed_deploy, chat_deploy


# ─────────────────────────────────────────────────────────────────────────────
# Query Refinement
# ─────────────────────────────────────────────────────────────────────────────

QUERY_REFINEMENT_PROMPT = """
You are a query classifier for the ABOG (American Board of Obstetrics and Gynecology)
board certification documentation system.

Given a user question (plus optional prior conversation context), rewrite it as
an optimised, self-contained search query and extract routing metadata.

The index contains pages from the ABOG website covering:
- Specialty certification: OB-GYN initial certification, Qualifying Exam, Certifying Exam
- Subspecialty certification: MFM, REI, Female Pelvic Medicine, Gynecologic Oncology, CCCS
- Maintenance of Certification (MOC) and CME requirements
- Dates, fees, eligibility bulletins, FAQs
- Alternate pathways to certification

Return a JSON object with exactly these fields:
{
  "detected_section": string or null,
  "detected_page_type": "policy" | "faq" | "overview" | "form" | null,
  "is_procedural": boolean,
  "standalone_query": string,
  "reasoning": string
}

Rules:
- If about exam steps, eligibility, case list → detected_section = "specialty-certification"
- If about MOC or renewal                     → detected_section = "moc"
- If about fees / deadlines / dates           → detected_section = "fees"
- If about subspecialties                     → detected_section = "subspecialty"
- standalone_query: fully rewrite the question so it stands alone, incorporating
  any relevant context from the conversation history.
"""


def refine_query(
    question: str,
    history: list[dict],
    openai_client,
    chat_deploy: str,
) -> tuple[str, dict]:
    """
    Returns (standalone_query, metadata_dict).
    Falls back to original question on any error.
    """
    history_text = ""
    if history:
        recent = history[-4:]
        history_text = "\n\nRecent conversation:\n" + "\n".join(
            f"{t['role'].upper()}: {t['content'][:200]}" for t in recent
        )

    messages = [
        {"role": "system", "content": QUERY_REFINEMENT_PROMPT},
        {
            "role": "user",
            "content": (
                f"User question: {question}"
                f"{history_text}\n\n"
                "Return only the JSON object described above."
            ),
        },
    ]

    try:
        t0 = time.time()
        response = openai_client.chat.completions.create(
            model=chat_deploy,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        elapsed = round(time.time() - t0, 2)
        raw     = response.choices[0].message.content.strip()
        raw     = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
        meta    = json.loads(raw)
        standalone = meta.get("standalone_query", question).strip().strip('"\'')
        meta["_latency_s"] = elapsed
        return (standalone if standalone else question), meta
    except Exception:
        return question, {}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Vector embedding search
# ─────────────────────────────────────────────────────────────────────────────

def embed_query(text: str, client, deploy: str) -> tuple[list[float], float]:
    """Returns (embedding_vector, latency_seconds)."""
    t0   = time.time()
    resp = client.embeddings.create(input=[text], model=deploy)
    return resp.data[0].embedding, round(time.time() - t0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Hybrid BM25 + Vector + Azure Semantic Reranker (recommended)
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_semantic_search(
    query: str,
    search_client,
    openai_client,
    embed_deploy: str,
    top_k: int = 7,
    odata_filter: Optional[str] = None,
) -> tuple[list[dict], dict]:
    """
    Full 3-stage pipeline:
      1. BM25 keyword match over text fields
      2. kNN vector search over content_vector field
      3. Azure Semantic Reranker fuses both via Reciprocal Rank Fusion (RRF)
         then re-scores with a cross-attention transformer model

    Score fields on each hit:
      _bm25_score     — pre-rerank @search.score (BM25+vector fusion score)
      _semantic_score — @search.reranker_score (0–4 scale, from Azure semantic model)
      _vector_score   — None (merged before semantic stage by RRF)
      _caption        — extractive highlight from the semantic model

    Returns (hits, embedding_diagnostics).
    """
    from azure.search.documents.models import VectorizedQuery

    vector, embed_latency = embed_query(query, openai_client, embed_deploy)
    vec_norm   = round(sum(v * v for v in vector) ** 0.5, 4)
    vec_dims   = len(vector)
    vec_sample = [round(v, 5) for v in vector[:5]]

    vq = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top_k * 2,
        fields="content_vector",
    )

    results = search_client.search(
        search_text=query,
        vector_queries=[vq],
        search_fields=[
            "page_title", "section", "breadcrumb",
            "section_h2", "section_h3", "content",
        ],
        query_type="semantic",
        semantic_configuration_name="semantic-config",
        query_caption="extractive",
        query_answer="extractive",
        top=top_k,
        filter=odata_filter,
        select=[
            "id", "chunk_index", "chunk_total",
            "url", "source_file", "page_title", "page_type",
            "section", "chunk_type", "section_h2", "section_h3",
            "breadcrumb", "content",
            "has_table", "is_callout", "contains_dates",
        ],
    )

    hits = []
    for r in results:
        d = dict(r)
        captions = r.get("@search.captions", [])
        if captions:
            d["_caption"] = captions[0].text
        d["_bm25_score"]     = round(r.get("@search.score", 0.0), 4)
        d["_semantic_score"] = (
            round(r["@search.reranker_score"], 4)
            if r.get("@search.reranker_score") is not None else None
        )
        d["_vector_score"]   = None
        hits.append(d)

    diag = {
        "model":           embed_deploy,
        "dimensions":      vec_dims,
        "l2_norm":         vec_norm,
        "sample_dims":     vec_sample,
        "embed_latency_s": embed_latency,
    }
    return hits, diag



# ─────────────────────────────────────────────────────────────────────────────
# GPT RAG — context builder + answer generation
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an ABOG (American Board of Obstetrics and Gynecology) Certification Assistant.
Your role is to provide accurate, helpful guidance about ABOG board certification
processes, requirements, exams, fees, and policies for OB-GYN physicians.

Your answers must prioritize accuracy. Only answer based on the source chunks provided.
Never fabricate requirements, dates, fees, or policy details.

---------------------------------------------------------------------
COVERAGE AREAS

• Specialty Certification (OB-GYN)
  - Five-step process: medical degree → accredited residency → Qualifying Exam
    → case list preparation → Certifying Exam
  - Eligibility and admissibility requirements
  - Case list preparation for the Certifying Exam
  - Exam formats, content outlines, scoring, and pass rates

• Subspecialty Certification
  - Maternal-Fetal Medicine (MFM)
  - Reproductive Endocrinology and Infertility (REI)
  - Female Pelvic Medicine and Reconstructive Surgery
  - Gynecologic Oncology
  - Complex Family Planning and Contraceptive Care (CCCS)

• Maintenance of Certification (MOC)
  - CME requirements, renewal cycles, assessment requirements

• Administrative
  - Dates, deadlines, fees
  - Alternate pathways to certification
  - Bulletins and FAQs

---------------------------------------------------------------------
RESPONSE FORMAT

1. Lead with a concise paragraph explanation.
2. Use numbered steps for multi-step processes.
3. Use bullet points for lists of requirements or conditions.
4. Use a table only when comparing exams, subspecialties, fees, or timelines.
5. Keep responses focused. Do not pad with unnecessary caveats.

---------------------------------------------------------------------
ALWAYS NOTE

It is the candidate's responsibility to verify current requirements directly
at abog.org and in the current candidate bulletin, as requirements can change.
"""


def build_context(hits: list[dict]) -> str:
    """
    Build LLM context. Policy/overview pages appear first to anchor the answer.
    """
    def priority(h):
        pt = (h.get("page_type")  or "").lower()
        ct = (h.get("chunk_type") or "").lower()
        if pt in ("policy", "overview"): return 0
        if ct == "table":                return 1
        return 2

    parts = []
    for i, h in enumerate(sorted(hits, key=priority), 1):
        title      = h.get("page_title", "")
        breadcrumb = h.get("breadcrumb", "")
        h2         = h.get("section_h2", "")
        h3         = h.get("section_h3", "")
        url        = h.get("url", "")
        ctype      = (h.get("chunk_type") or "text").upper()
        flags = []
        if h.get("has_table"):      flags.append("HAS_TABLE")
        if h.get("is_callout"):     flags.append("CALLOUT")
        if h.get("contains_dates"): flags.append("DATES")
        heading  = " > ".join(filter(None, [h2, h3]))
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        parts.append(
            f"[SOURCE {i}] [{ctype}]{flag_str}  {title}\n"
            f"Breadcrumb : {breadcrumb}\n"
            f"Sub-section: {heading}\n"
            f"URL        : {url}\n\n"
            f"{h['content']}\n"
        )
    return "\n---\n".join(parts)


def generate_answer(
    question: str,
    hits: list[dict],
    history: list[dict],
    openai_client,
    chat_deploy: str,
) -> tuple[str, float]:
    context  = build_context(hits)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": (
            "Use the following source chunks to answer the question.\n\n"
            f"=== SOURCE CHUNKS ===\n{context}\n\n"
            f"=== QUESTION ===\n{question}"
        ),
    })

    t0 = time.time()
    response = openai_client.chat.completions.create(
        model=chat_deploy,
        messages=messages,
        temperature=0.1,
        max_tokens=1200,
    )
    return response.choices[0].message.content.strip(), round(time.time() - t0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_badge(page_type: str, chunk_type: str) -> tuple[str, str]:
    ct = (chunk_type or "").lower()
    pt = (page_type  or "").lower()
    if ct == "table":       return "TABLE",   "badge-table"
    if pt == "policy":      return "POLICY",  "badge-policy"
    if pt == "faq":         return "FAQ",     "badge-faq"
    if "callout" in ct:     return "CALLOUT", "badge-callout"
    return "TEXT", "badge-text"


def render_source_card(hit: dict, index: int):
    page_type  = hit.get("page_type",  "")
    chunk_type = hit.get("chunk_type", "text")
    label, cls = _chunk_badge(page_type, chunk_type)

    title      = hit.get("page_title", "—")
    breadcrumb = hit.get("breadcrumb", "")
    url        = hit.get("url", "")
    caption    = hit.get("_caption", "")

    bm25     = hit.get("_bm25_score")
    vec      = hit.get("_vector_score")
    semantic = hit.get("_semantic_score")
    gpt_r    = hit.get("_gpt_rerank_score")

    flags = []
    if hit.get("has_table"):      flags.append("📊 Table")
    if hit.get("is_callout"):     flags.append("📌 Callout")
    if hit.get("contains_dates"): flags.append("📅 Dates")

    link_html = (
        f'<a class="doc-link" href="{url}" target="_blank">{title}</a>'
        if url else
        f'<span class="doc-link">{title}</span>'
    )

    # Score pills — only render stages that ran
    score_pills = ""
    if bm25     is not None:
        score_pills += f'<span class="score-pill pill-bm25">BM25 {bm25:.3f}</span>'
    if vec      is not None:
        score_pills += f'<span class="score-pill pill-vector">Vector {vec:.3f}</span>'
    if semantic is not None:
        score_pills += f'<span class="score-pill pill-semantic">Semantic {semantic:.3f}</span>'
    if gpt_r    is not None:
        score_pills += f'<span class="score-pill pill-gpt">GPT {gpt_r:.2f}</span>'

    flag_html = (
        f'<div style="font-size:0.76rem; color:#6b7280; margin-top:3px;">'
        f'{" &nbsp; ".join(flags)}</div>'
        if flags else ""
    )

    st.markdown(f"""
    <div class="source-card">
        <div>
            {link_html}
            <span class="badge {cls}">{label}</span>
        </div>
        <div class="breadcrumb-text">🗂 {breadcrumb}</div>
        <div style="margin-top:5px;">{score_pills}</div>
        {flag_html}
        {f'<div style="color:#4b5563; font-size:0.82rem; font-style:italic; margin-top:5px;">"{caption}"</div>' if caption else ""}
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"View chunk {index} content"):
        st.markdown(hit.get("content", ""), unsafe_allow_html=False)


def render_pipeline_bar(stages_active: list[str]):
    """Render a horizontal pipeline indicator showing active stages."""
    all_stages = [
        "BM25 Keyword",
        "Vector Embedding",
        "Semantic Reranker",
        "GPT Rerank",
        "LLM Generation",
    ]
    pills = []
    for s in all_stages:
        cls = "pipeline-stage active" if s in stages_active else "pipeline-stage"
        pills.append(f'<span class="{cls}">{s}</span>')
    inner = ' <span class="pipeline-arrow">→</span> '.join(pills)
    st.markdown(f'<div class="pipeline-row">{inner}</div>', unsafe_allow_html=True)


def render_embed_diagnostics(diag: dict):
    """Display embedding diagnostics in a collapsible panel."""
    with st.expander("🔬 Embedding diagnostics", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model",       diag.get("model", "—"))
        c2.metric("Dimensions",  diag.get("dimensions", "—"))
        c3.metric("L2 Norm",     diag.get("l2_norm", "—"))
        c4.metric("Embed (s)",   diag.get("embed_latency_s", "—"))
        st.caption(f"First 5 dims: `{diag.get('sample_dims', [])}`")


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

for _k, _v in [
    ("messages",        []),
    ("total_queries",   0),
    ("total_sources",   0),
    ("disclaimer_shown", False),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 ABOG Assistant")
    st.caption("Board Certification Knowledge Base")
    st.markdown("---")

    search_client, openai_client, embed_deploy, chat_deploy = get_clients()

    # ── Search mode ────────────────────────────────────────────────────────
    st.markdown("### 🔍 Search Pipeline")
    search_mode = st.radio(
        "Active search stages",
        options=[
            "Hybrid + Semantic Reranker  ✦ recommended",
            "Vector only",
            "BM25 keyword only",
        ],
        index=0,
        help=(
            "**Hybrid + Semantic** (recommended): BM25 + vector fusion via "
            "Reciprocal Rank Fusion → Azure Semantic Reranker "
            "(cross-attention transformer, scores 0–4).\n\n"
            "**Vector only**: pure kNN cosine on content_vector field.\n\n"
            "**BM25 only**: lexical keyword match, fastest but no semantic understanding."
        ),
    )

    st.markdown("---")
    st.markdown("### ⚙️ Retrieval Settings")

    top_k = st.slider("Chunks to retrieve", 3, 20, 15)
    top_n_rerank = st.slider(
        "Chunks after GPT rerank",
        3, 15, 8,
        help="GPT cross-encoder keeps only the top-N most relevant chunks.",
    )

    filter_type = st.selectbox(
        "Filter by page type",
        ["All", "policy", "faq", "overview", "form"],
    )
    filter_section = st.text_input(
        "Filter by section slug",
        placeholder="e.g. get-certified",
    )

    st.markdown("---")
    st.markdown("### 🔧 Optional Steps")

    enable_refinement = st.toggle(
        "Query refinement",
        value=True,
        help="LLM rewrites the question into an optimised standalone search query.",
    )
    enable_gpt_rerank = st.toggle(
        "GPT cross-encoder rerank",
        value=True,
        help=(
            "After Azure search, GPT scores each chunk 0–1 against the "
            "original question and keeps the top-N most relevant."
        ),
    )
    show_embed_diag = st.toggle(
        "Show embedding diagnostics",
        value=False,
        help="Displays vector dimensions, L2 norm, sample dims, and latency.",
    )

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{st.session_state.total_queries}</div>
            <div class="metric-label">Queries</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{st.session_state.total_sources}</div>
            <div class="metric-label">Sources</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        for k in ["messages", "total_queries", "total_sources"]:
            st.session_state[k] = [] if k == "messages" else 0
        st.session_state.disclaimer_shown = False
        st.rerun()

    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "What are the 5 steps to OB-GYN board certification?",
        "What is required to apply for the Qualifying Exam?",
        "How is the case list prepared for the Certifying Exam?",
        "What subspecialties does ABOG certify?",
        "How do I maintain my ABOG certification (MOC)?",
        "Is there an alternate pathway to certification?",
        "What are the current Certifying Exam fees and dates?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:22]}"):
            st.session_state._pending_question = s
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🏥 ABOG Certification Assistant")
st.markdown(
    "Ask questions about ABOG board certification in obstetrics & gynecology — "
    "exam steps, eligibility, case lists, subspecialties, MOC, fees, and policies."
)
st.markdown("---")

# ── Render conversation history ────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🏥"):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            orig    = msg.get("original_query", "")
            refined = msg.get("refined_query", "")
            if orig and refined and refined != orig:
                st.markdown(
                    f'<div class="refined-query-banner">'
                    f'<strong>🔍 Refined query:</strong> {refined}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if msg.get("hits"):
                with st.expander(f"📚 {len(msg['hits'])} source chunks used", expanded=False):
                    for i, hit in enumerate(msg["hits"], 1):
                        render_source_card(hit, i)

            if msg.get("timings"):
                t = msg["timings"]
                st.caption(
                    f"⏱ Search {t.get('search_s', '?')}s · "
                    f"Generation {t.get('gen_s', '?')}s"
                )


# ── Chat input ─────────────────────────────────────────────────────────────

pending = getattr(st.session_state, "_pending_question", None)
if pending:
    del st.session_state._pending_question
    user_input = pending
else:
    user_input = st.chat_input(
        "Ask about ABOG exams, eligibility, MOC, fees, subspecialties…",
        disabled=(search_client is None),
    )

if user_input:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="🏥"):
        answer_ph  = st.empty()
        refined_ph = st.empty()
        source_ph  = st.empty()

        plain_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        # ── Determine active pipeline stages ──────────────────────────────
        _mode = search_mode.split("✦")[0].strip()
        active_stages = ["LLM Generation"]
        if "Hybrid"  in _mode: active_stages += ["BM25 Keyword", "Vector Embedding", "Semantic Reranker"]
        elif "Vector" in _mode: active_stages += ["Vector Embedding"]
        elif "BM25"   in _mode: active_stages += ["BM25 Keyword"]
        if enable_gpt_rerank:  active_stages.append("GPT Rerank")
        render_pipeline_bar(active_stages)

        # ── OData filter ───────────────────────────────────────────────────
        filters = []
        if filter_type and filter_type != "All":
            filters.append(f"page_type eq '{filter_type}'")
        if filter_section.strip():
            safe = filter_section.strip().replace("'", "''")
            filters.append(f"section eq '{safe}'")
        odata_filter = " and ".join(filters) if filters else None

        # ── Step 1: Query refinement ───────────────────────────────────────
        refined_query = user_input
        refine_meta   = {}
        if enable_refinement and openai_client:
            with st.spinner("Refining query…"):
                refined_query, refine_meta = refine_query(
                    question=user_input,
                    history=plain_history,
                    openai_client=openai_client,
                    chat_deploy=chat_deploy,
                )
            if refined_query != user_input:
                refined_ph.markdown(
                    f'<div class="refined-query-banner">'
                    f'<strong>🔍 Refined query:</strong> {refined_query}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Step 2: Search ─────────────────────────────────────────────────
        embed_diag = {}
        t_search   = time.time()
        hits       = []

        try:
            # if "Hybrid" in _mode:
            with st.spinner(
                "Stage 1 BM25 keyword match → Stage 2 vector embedding → "
                "Stage 3 Azure Semantic Reranker…"
            ):
                hits, embed_diag = hybrid_semantic_search(
                    query=refined_query,
                    search_client=search_client,
                    openai_client=openai_client,
                    embed_deploy=embed_deploy,
                    top_k=top_k,
                    odata_filter=odata_filter,
                )
        except Exception as e:
            st.error(f"Search error: {e}")
            hits = []

        search_elapsed = round(time.time() - t_search, 2)

        # Embedding diagnostics panel
        if show_embed_diag and embed_diag:
            render_embed_diagnostics(embed_diag)

        # ── Step 4: Generate answer ────────────────────────────────────────
        if not hits:
            answer      = (
                "I couldn't find relevant information for that question in the "
                "indexed documents. Try rephrasing, or verify the documents have "
                "been indexed correctly."
            )
            gen_elapsed = 0.0
        else:
            with st.spinner("Generating answer…"):
                try:
                    answer, gen_elapsed = generate_answer(
                        question=user_input,
                        hits=hits,
                        history=plain_history,
                        openai_client=openai_client,
                        chat_deploy=chat_deploy,
                    )
                except Exception as e:
                    answer      = f"Generation error: {e}"
                    gen_elapsed = 0.0

        # ── First-response disclaimer ──────────────────────────────────────
        if not st.session_state.disclaimer_shown:
            st.markdown(
                '<div class="disclaimer">'
                '<strong>⚠️ Disclaimer:</strong> '
                'This assistant provides guidance based on indexed ABOG documentation. '
                'Certification requirements, fees, and deadlines can change — always '
                'verify current requirements at '
                '<a href="https://www.abog.org" target="_blank" style="color:#b45309;">'
                'abog.org</a> and review the current candidate bulletin.'
                '</div>',
                unsafe_allow_html=True,
            )
            st.session_state.disclaimer_shown = True

        answer_ph.markdown(answer)

        if hits:
            with source_ph.expander(
                f"📚 {len(hits)} source chunks used", expanded=False
            ):
                for i, hit in enumerate(hits, 1):
                    render_source_card(hit, i)

        timing_parts = [
            f"⏱ Search {search_elapsed}s",
            f"Generation {gen_elapsed}s",
        ]
        if refine_meta.get("_latency_s"):
            timing_parts.append(f"Refinement {refine_meta['_latency_s']}s")
        st.caption(" · ".join(timing_parts))

    # ── Persist ───────────────────────────────────────────────────────────
    st.session_state.messages.append({
        "role":           "assistant",
        "content":        answer,
        "hits":           hits,
        "original_query": user_input,
        "refined_query":  refined_query,
        "timings": {"search_s": search_elapsed, "gen_s": gen_elapsed},
    })
    st.session_state.total_queries += 1
    st.session_state.total_sources += len(hits)