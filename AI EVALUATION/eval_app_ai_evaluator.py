"""
eval_app.py  —  RAG Evaluator Streamlit App (azure-ai-evaluation SDK)
----------------------------------------------------------------------
Uses the official azure-ai-evaluation library for 3 standard dimensions:
  ┌─────────────────────┬──────────────────────────────────────────────────┐
  │ Dimension           │ Evaluator                                        │
  ├─────────────────────┼──────────────────────────────────────────────────┤
  │ Groundedness        │ GroundednessEvaluator  (SDK built-in)            │
  │ Relevance           │ RelevanceEvaluator     (SDK built-in)            │
  │ Response Complete.  │ ResponseCompletenessEvaluator (SDK built-in)     │
  │ Retrieval Quality   │ Custom LLM-as-judge    (SDK can't do chunk-level)│
  └─────────────────────┴──────────────────────────────────────────────────┘

SDK scores are on a 1–5 Likert scale. We normalise to 0–1 for the
composite score so all dimensions are on the same footing.

Install:
  pip install azure-ai-evaluation

Environment variables (same .env as chatbot_app.py):
  AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
  AZURE_OPENAI_EMBED_DEPLOY, AZURE_OPENAI_CHAT_DEPLOY
  SEARCH_INDEX_NAME

Run:
  streamlit run eval_app.py
"""

import json
import os
import re
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config & styling
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Evaluator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #080c14; }
[data-testid="stSidebar"] { background: #0c1120; border-right: 1px solid #141d2e; }

/* ── Score cards ── */
.dim-card {
    background: #0e1628; border: 1px solid #1a2540;
    border-radius: 12px; padding: 20px 16px 16px;
    text-align: center; position: relative; overflow: hidden;
}
.dim-card .accent-bar { position: absolute; top:0; left:0; right:0; height:2px; }
.dim-card .dim-score  { font-family:'DM Mono',monospace; font-size:2.4rem; font-weight:500; line-height:1; margin:6px 0 4px; }
.dim-card .dim-name   { font-size:.68rem; font-weight:600; text-transform:uppercase; letter-spacing:.1em; color:#4a5568; margin-bottom:4px; }
.dim-card .dim-raw    { font-size:.65rem; color:#2d3748; font-family:'DM Mono',monospace; }
.dim-card .dim-badge  { display:inline-block; padding:1px 8px; border-radius:10px; font-size:.65rem; font-weight:700; margin-top:5px; }

/* score colours */
.score-hi   { color:#34d399; }
.score-mid  { color:#fbbf24; }
.score-lo   { color:#f87171; }
.score-none { color:#374151; }

.bar-green  { background:linear-gradient(90deg,#34d399,#059669); }
.bar-amber  { background:linear-gradient(90deg,#fbbf24,#d97706); }
.bar-red    { background:linear-gradient(90deg,#f87171,#dc2626); }
.bar-gray   { background:#1f2d44; }
.bar-blue   { background:linear-gradient(90deg,#60a5fa,#3b82f6); }

.badge-pass { background:#064e3b; color:#34d399; }
.badge-fail { background:#450a0a; color:#f87171; }
.badge-na   { background:#1f2d44; color:#4b5563; }

/* ── Composite ── */
.composite-wrap {
    background:#0e1628; border:1px solid #1a2540;
    border-radius:12px; padding:20px; text-align:center;
}
.composite-val   { font-family:'Syne',sans-serif; font-size:3.2rem; font-weight:800; line-height:1; }
.composite-label { font-size:.68rem; text-transform:uppercase; letter-spacing:.14em; color:#4a5568; font-weight:600; margin-top:6px; }

/* ── SDK badge ── */
.sdk-tag {
    display:inline-block; background:#1e3a6e; color:#60a5fa;
    border:1px solid #1d4ed8; border-radius:4px;
    padding:1px 7px; font-size:.65rem; font-weight:700;
    letter-spacing:.06em; margin-left:6px; vertical-align:middle;
}
.custom-tag {
    display:inline-block; background:#3b2a6e; color:#a78bfa;
    border:1px solid #5b21b6; border-radius:4px;
    padding:1px 7px; font-size:.65rem; font-weight:700;
    letter-spacing:.06em; margin-left:6px; vertical-align:middle;
}

/* ── Answer panes ── */
.answer-pane { background:#080c14; border:1px solid #141d2e; border-radius:8px; padding:14px 16px; }
.pane-label  { font-size:.65rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid #141d2e; }
.pane-label.gt  { color:#34d399; }
.pane-label.gen { color:#60a5fa; }
.pane-text   { font-size:.84rem; color:#94a3b8; line-height:1.6; white-space:pre-wrap; }

/* ── Reasoning box ── */
.detail-box { background:#080c14; border:1px dashed #1a2540; border-radius:8px; padding:12px 16px; font-size:.82rem; color:#64748b; line-height:1.6; margin-top:8px; }
.detail-box strong { color:#94a3b8; }

/* ── Issue / ok items ── */
.issue-item { background:#120a0a; border-left:2px solid #f87171; border-radius:0 6px 6px 0; padding:6px 12px; font-size:.8rem; color:#fca5a5; margin:4px 0; }
.ok-item    { background:#061412; border-left:2px solid #34d399; border-radius:0 6px 6px 0; padding:6px 12px; font-size:.8rem; color:#6ee7b7; margin:4px 0; }

/* ── Pills ── */
.pill      { display:inline-flex; align-items:center; gap:4px; padding:3px 10px; border-radius:20px; font-family:'DM Mono',monospace; font-size:.7rem; font-weight:500; margin:0 4px 4px 0; }
.pill-hi   { background:#064e3b; color:#34d399; border:1px solid #065f46; }
.pill-mid  { background:#451a03; color:#fbbf24; border:1px solid #78350f; }
.pill-lo   { background:#450a0a; color:#f87171; border:1px solid #7f1d1d; }
.pill-blue { background:#1e3a6e; color:#60a5fa; border:1px solid #1d4ed8; }

/* ── Source chips ── */
.src-chip { display:inline-block; background:#0e1628; border:1px solid #1a2540; color:#64748b; border-radius:4px; padding:2px 8px; font-size:.68rem; font-family:'DM Mono',monospace; margin:2px 3px 2px 0; }

/* ── Misc ── */
.sec-label { font-size:.65rem; text-transform:uppercase; letter-spacing:.14em; color:#2d3748; font-weight:700; margin:20px 0 10px; padding-bottom:6px; border-bottom:1px solid #111827; }
.prog-pill { background:#0e1628; border:1px solid #1a2540; border-radius:8px; padding:10px 16px; font-family:'DM Mono',monospace; font-size:.78rem; color:#60a5fa; margin:4px 0; }
.empty-state { background:#0e1628; border:1px dashed #1a2540; border-radius:14px; padding:48px 24px; text-align:center; }
.empty-state .es-icon { font-size:2.4rem; margin-bottom:10px; }
.empty-state .es-text { font-family:'Syne',sans-serif; font-size:1rem; color:#374151; }

h1, h2, h3 { font-family:'Syne',sans-serif !important; color:#f1f5f9 !important; }
p, li { color:#64748b; }
.stTextArea textarea, .stTextInput input { background:#080c14 !important; border-color:#1a2540 !important; color:#e2e8f0 !important; }
.stButton > button { font-family:'Syne',sans-serif !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Azure clients (cached)
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

    search_client = SearchClient(endpoint, index_name, AzureKeyCredential(key))
    openai_client = AzureOpenAI(
        azure_endpoint=aoai_ep, api_key=aoai_key, api_version="2024-02-01"
    )
    embed_deploy = os.environ.get("AZURE_OPENAI_EMBED_DEPLOY", "text-embedding-3-small")
    chat_deploy  = os.environ.get("AZURE_OPENAI_CHAT_DEPLOY",  "gpt-4o-mini")
    return search_client, openai_client, embed_deploy, chat_deploy


# ─────────────────────────────────────────────────────────────────────────────
# azure-ai-evaluation SDK model config (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading evaluators…")
def get_sdk_evaluators():
    """
    Initialise the three built-in SDK evaluators once and reuse across calls.
    Each evaluator needs an AzureOpenAIModelConfiguration pointing at your
    GPT deployment — the same one used for generation.

    SDK scores are on a 1–5 Likert scale.
    Pass threshold default = 3 (i.e. score >= 3 → pass).
    """
    try:
        from azure.ai.evaluation import (
            AzureOpenAIModelConfiguration,
            GroundednessEvaluator,
            RelevanceEvaluator,
            ResponseCompletenessEvaluator,
        )

        model_config = AzureOpenAIModelConfiguration(
            azure_endpoint  = os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_key         = os.environ.get("AZURE_OPENAI_API_KEY", ""),
            azure_deployment= os.environ.get("AZURE_OPENAI_CHAT_DEPLOY", "gpt-4o-mini"),
            api_version     = "2024-02-01",
        )

        return {
            "groundedness":           GroundednessEvaluator(model_config=model_config),
            "relevance":              RelevanceEvaluator(model_config=model_config),
            "response_completeness":  ResponseCompletenessEvaluator(model_config=model_config),
        }
    except ImportError:
        return None  # handled gracefully in the UI
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RAG pipeline helpers (same as chatbot_app.py)
# ─────────────────────────────────────────────────────────────────────────────

def embed_query(text, client, deploy):
    return client.embeddings.create(input=[text], model=deploy).data[0].embedding


def hybrid_search(query, search_client, openai_client, embed_deploy, top_k=15):
    from azure.search.documents.models import VectorizedQuery
    vq = VectorizedQuery(
        vector=embed_query(query, openai_client, embed_deploy),
        k_nearest_neighbors=top_k * 2,
        fields="content_vector",
    )
    results = search_client.search(
        search_text=query,
        vector_queries=[vq],
        search_fields=["section_title", "topic", "subtopic", "content"],
        query_type="semantic",
        semantic_configuration_name="semantic-config",
        query_caption="extractive",
        top=top_k,
        select=[
            "chunk_id", "source", "source_pdf_url", "source_pdf_name",
            "section_title", "section", "topic", "subtopic",
            "content", "content_type", "page_start", "page_end",
        ],
    )
    hits = []
    for r in results:
        d = dict(r)
        d["_score"] = r.get("@search.reranker_score") or r.get("@search.score", 0)
        hits.append(d)
    return hits


def build_context(hits):
    """NY ESCO Doc chunks first (Tier 1), then utility-specific chunks."""
    def tier(h):
        n = (h.get("source_pdf_name") or h.get("source") or "").lower()
        return 0 if ("esco doc" in n or "esco operating" in n) else 1
    parts = []
    for i, h in enumerate(sorted(hits, key=tier), 1):
        ps, pe = h.get("page_start"), h.get("page_end")
        pg = f"(Pages {ps}–{pe})" if ps and pe and ps != pe else (f"(Page {ps})" if ps else "")
        parts.append(
            f"[SOURCE {i}] [{h.get('content_type','text').upper()}] "
            f"{h.get('section', h.get('section_title',''))} {pg}\n{h['content']}\n"
        )
    return "\n---\n".join(parts)


GENERATION_SYSTEM = """You are a Retail Energy Regulatory Assistant.
Answer questions accurately using ONLY the provided source chunks.
Never fabricate information not present in the sources."""

def generate_answer(question, hits, openai_client, chat_deploy):
    resp = openai_client.chat.completions.create(
        model=chat_deploy,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM},
            {"role": "user",   "content": f"Source chunks:\n{build_context(hits)}\n\nQuestion: {question}"},
        ],
        temperature=0.1, max_tokens=1200,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# SDK evaluators  (Groundedness / Relevance / ResponseCompleteness)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(raw_score):
    """Convert 1–5 Likert → 0–1 float, rounded to 2dp."""
    try:
        return round((float(raw_score) - 1) / 4, 2)
    except Exception:
        return None


def run_sdk_groundedness(evaluators, question, generated, context_str):
    """
    GroundednessEvaluator signature: (query, response, context)
    Returns raw 1-5 score, normalised 0-1 score, reason string, pass/fail.
    """
    try:
        result = evaluators["groundedness"](
            query=question,
            response=generated,
            context=context_str,
        )
        raw    = result.get("groundedness") or result.get("gpt_groundedness")
        reason = result.get("groundedness_reason", "")
        passed = result.get("groundedness_result", "")   # "pass" | "fail"
        return {
            "source":     "azure-ai-evaluation SDK",
            "raw_score":  raw,
            "score":      _normalise(raw),
            "reason":     reason,
            "passed":     passed,
        }
    except Exception as e:
        return {"source": "azure-ai-evaluation SDK", "raw_score": None, "score": None,
                "reason": str(e), "passed": "error"}


def run_sdk_relevance(evaluators, question, generated):
    """
    RelevanceEvaluator signature: (query, response)
    """
    try:
        result = evaluators["relevance"](
            query=question,
            response=generated,
        )
        raw    = result.get("relevance") or result.get("gpt_relevance")
        reason = result.get("relevance_reason", "")
        passed = result.get("relevance_result", "")
        return {
            "source":     "azure-ai-evaluation SDK",
            "raw_score":  raw,
            "score":      _normalise(raw),
            "reason":     reason,
            "passed":     passed,
        }
    except Exception as e:
        return {"source": "azure-ai-evaluation SDK", "raw_score": None, "score": None,
                "reason": str(e), "passed": "error"}


def run_sdk_completeness(evaluators, generated, ground_truth):
    """
    ResponseCompletenessEvaluator signature: (response, ground_truth)
    This replaces our old "correctness" dimension — it measures recall vs ground truth.
    """
    try:
        result = evaluators["response_completeness"](
            response=generated,
            ground_truth=ground_truth,
        )
        raw    = result.get("response_completeness") or result.get("gpt_response_completeness")
        reason = result.get("response_completeness_reason", "")
        passed = result.get("response_completeness_result", "")
        return {
            "source":     "azure-ai-evaluation SDK",
            "raw_score":  raw,
            "score":      _normalise(raw),
            "reason":     reason,
            "passed":     passed,
        }
    except Exception as e:
        return {"source": "azure-ai-evaluation SDK", "raw_score": None, "score": None,
                "reason": str(e), "passed": "error"}


# ─────────────────────────────────────────────────────────────────────────────
# Custom retrieval evaluator  (SDK has no chunk-level scorer)
# ─────────────────────────────────────────────────────────────────────────────

def run_custom_retrieval(question, hits, openai_client, chat_deploy):
    """
    LLM-as-judge for retrieval quality — scores each chunk 0–1 and gives
    an overall retrieval score. This is the one dimension the SDK can't cover
    at the chunk level, so we keep our own evaluator here.
    """
    EVAL_SYSTEM = """You are a strict retrieval quality evaluator.
Respond ONLY with a valid JSON object. No preamble, no markdown fences."""

    chunks_text = "".join(
        f"\n[CHUNK {i}] Source: {h.get('source_pdf_name','?')}\n{h.get('content','')[:300]}\n"
        for i, h in enumerate(hits[:8])
    )
    prompt = f"""Evaluate RETRIEVAL QUALITY for this question.
Score each retrieved chunk's relevance (0.0–1.0) and give an overall score.

Question: {question}
Retrieved Chunks:\n{chunks_text}

Return JSON:
{{
  "overall_score": <0.0–1.0>,
  "reasoning": "<one paragraph>",
  "chunk_scores": [0.9, 0.4, ...],
  "top_chunk_is_relevant": true,
  "irrelevant_chunks_count": 2
}}

1.0=all chunks highly relevant, 0.7=mostly relevant, 0.4=mixed noise, 0.0=entirely off-topic"""

    try:
        raw = openai_client.chat.completions.create(
            model=chat_deploy,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0, max_tokens=400,
        ).choices[0].message.content.strip()
        raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
        result = json.loads(raw)
        return {
            "source":                 "custom LLM-as-judge",
            "score":                  result.get("overall_score"),
            "reasoning":              result.get("reasoning", ""),
            "chunk_scores":           result.get("chunk_scores", []),
            "top_chunk_is_relevant":  result.get("top_chunk_is_relevant"),
            "irrelevant_chunks_count":result.get("irrelevant_chunks_count"),
        }
    except Exception as e:
        return {"source": "custom LLM-as-judge", "score": None, "reasoning": str(e),
                "chunk_scores": [], "top_chunk_is_relevant": None, "irrelevant_chunks_count": None}


# ─────────────────────────────────────────────────────────────────────────────
# Composite score
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "groundedness":          0.30,
    "relevance":             0.25,
    "response_completeness": 0.30,
    "retrieval_quality":     0.15,
}

def composite_score(ev):
    ws = wt = 0.0
    for dim, w in WEIGHTS.items():
        s = ev.get(dim, {}).get("score")
        if s is not None:
            ws += s * w; wt += w
    return round(ws / wt, 4) if wt else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_single_eval(
    question, ground_truth,
    search_client, openai_client, embed_deploy, chat_deploy,
    sdk_evaluators, top_k, delay, status_fn=None
):
    def step(msg):
        if status_fn: status_fn(msg)

    # 1 — Retrieve
    step("🔍 Retrieving chunks from Azure AI Search…")
    hits = hybrid_search(question, search_client, openai_client, embed_deploy, top_k)
    context_str = build_context(hits)
    time.sleep(delay)

    # 2 — Generate
    step(f"🤖 Generating answer ({len(hits)} chunks retrieved)…")
    generated = generate_answer(question, hits, openai_client, chat_deploy)
    time.sleep(delay)

    # 3 — SDK: Groundedness
    step("📐 [azure-ai-evaluation] Groundedness…")
    g  = run_sdk_groundedness(sdk_evaluators, question, generated, context_str)
    time.sleep(delay)

    # 4 — SDK: Relevance
    step("🎯 [azure-ai-evaluation] Relevance…")
    r  = run_sdk_relevance(sdk_evaluators, question, generated)
    time.sleep(delay)

    # 5 — SDK: Response Completeness (vs ground truth)
    step("✔️  [azure-ai-evaluation] Response Completeness vs your answer…")
    rc = run_sdk_completeness(sdk_evaluators, generated, ground_truth)
    time.sleep(delay)

    # 6 — Custom: Retrieval Quality
    step("🔎 [custom] Retrieval quality (chunk-level)…")
    rq = run_custom_retrieval(question, hits, openai_client, chat_deploy)
    time.sleep(delay)

    ev = {
        "groundedness":          g,
        "relevance":             r,
        "response_completeness": rc,
        "retrieval_quality":     rq,
    }
    ev["composite_score"] = composite_score(ev)

    return {
        "question":            question,
        "ground_truth_answer": ground_truth,
        "generated_answer":    generated,
        "context_str":         context_str,
        "retrieved_sources":   [h.get("source_pdf_name", h.get("source","")) for h in hits],
        "num_chunks":          len(hits),
        "eval":                ev,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def score_cls(s):
    if s is None:  return "score-none", "bar-gray"
    if s >= 0.75:  return "score-hi",   "bar-green"
    if s >= 0.50:  return "score-mid",  "bar-amber"
    return "score-lo", "bar-red"

def pill_cls(s):
    if s is None:  return "pill-blue"
    if s >= 0.75:  return "pill-hi"
    if s >= 0.50:  return "pill-mid"
    return "pill-lo"

def fmt(s):
    return f"{s:.2f}" if s is not None else "—"

def fmt_raw(raw):
    """Format raw 1-5 SDK score for display."""
    try:    return f"{float(raw):.1f}/5"
    except: return "—"

def passed_badge(passed_str):
    if passed_str == "pass":
        return '<span class="dim-badge badge-pass">PASS</span>'
    elif passed_str == "fail":
        return '<span class="dim-badge badge-fail">FAIL</span>'
    else:
        return '<span class="dim-badge badge-na">—</span>'

def source_tag(source_str):
    if "sdk" in source_str.lower():
        return '<span class="sdk-tag">SDK</span>'
    return '<span class="custom-tag">CUSTOM</span>'

def render_dim_card(label, score, source_str, raw=None, passed_str=None, weight_pct=None):
    sc, bc = score_cls(score)
    badge  = passed_badge(passed_str) if passed_str else ""
    stag   = source_tag(source_str)
    raw_display = f"<div class='dim-raw'>{fmt_raw(raw)}</div>" if raw is not None else ""
    wpct   = f"<div class='dim-raw'>weight {weight_pct}%</div>" if weight_pct else ""
    return f"""
<div class="dim-card">
  <div class="accent-bar {bc}"></div>
  <div class="dim-name">{label} {stag}</div>
  <div class="dim-score {sc}">{fmt(score)}</div>
  {raw_display}
  {badge}
  {wpct}
</div>"""

def render_composite(score):
    sc, _ = score_cls(score)
    return f"""
<div class="composite-wrap">
  <div class="composite-val {sc}">{fmt(score)}</div>
  <div class="composite-label">Composite Score</div>
</div>"""

def render_issues(items, kind="issue"):
    if not items: return ""
    cls = "issue-item" if kind == "issue" else "ok-item"
    return "".join(f'<div class="{cls}">{item}</div>' for item in items)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = []


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧪 RAG Evaluator")
    st.markdown("---")

    search_client, openai_client, embed_deploy, chat_deploy = get_clients()
    sdk_evaluators = get_sdk_evaluators()

    if not search_client:
        st.error("⚠️ Azure credentials missing. Check your .env file.")
    else:
        st.success("✅ Azure AI Search connected")

    if sdk_evaluators is None:
        st.warning("⚠️ `azure-ai-evaluation` not installed.\n\nRun:\n```\npip install azure-ai-evaluation\n```")
    else:
        st.success("✅ azure-ai-evaluation SDK ready")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", 5, 20, 15)
    delay = st.slider("API delay (s)", 0.5, 3.0, 1.0, 0.5,
                      help="Pause between evaluator calls to avoid rate limits")

    st.markdown("---")
    st.markdown("### 📐 Evaluator Map")

    evaluator_map = [
        ("Groundedness",         "#34d399", "30%", "SDK"),
        ("Relevance",            "#a78bfa", "25%", "SDK"),
        ("Resp. Completeness",   "#fbbf24", "30%", "SDK"),
        ("Retrieval Quality",    "#60a5fa", "15%", "CUSTOM"),
    ]
    for label, color, pct, src in evaluator_map:
        tag = f'<span class="sdk-tag">{src}</span>' if src == "SDK" else f'<span class="custom-tag">{src}</span>'
        st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            margin-bottom:6px;padding:6px 10px;background:#0e1628;
            border-radius:6px;border:1px solid #1a2540;">
  <span style="font-size:.78rem;color:#64748b;">{label} {tag}</span>
  <span style="font-family:'DM Mono',monospace;font-size:.78rem;color:{color};">{pct}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:#0c1120;border:1px solid #1a2540;border-radius:8px;
            padding:10px 12px;margin-top:8px;font-size:.75rem;color:#4a5568;">
  SDK scores are on a <strong style="color:#94a3b8">1–5 Likert scale</strong>
  normalised to 0–1 for the composite.
  Pass threshold = 3/5.
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    n_total = len(st.session_state.results)
    scores  = [r["eval"].get("composite_score") for r in st.session_state.results if r["eval"].get("composite_score")]
    avg     = round(sum(scores)/len(scores), 2) if scores else None
    sc_cls, _ = score_cls(avg)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
<div style="background:#0e1628;border:1px solid #1a2540;border-radius:8px;padding:12px;text-align:center;">
  <div style="font-family:'DM Mono',monospace;font-size:1.6rem;color:#60a5fa;">{n_total}</div>
  <div style="font-size:.65rem;color:#2d3748;text-transform:uppercase;letter-spacing:.1em;">Evaluated</div>
</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
<div style="background:#0e1628;border:1px solid #1a2540;border-radius:8px;padding:12px;text-align:center;">
  <div style="font-family:'DM Mono',monospace;font-size:1.6rem;" class="{sc_cls}">{fmt(avg)}</div>
  <div style="font-size:.65rem;color:#2d3748;text-transform:uppercase;letter-spacing:.1em;">Avg Score</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear all results", use_container_width=True):
        st.session_state.results = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🧪 RAG Evaluator")
st.markdown(
    "<p style='color:#374151;margin-top:-8px'>Enter a question and your expected answer — "
    "the app retrieves, generates, then scores via "
    "<span class='sdk-tag'>azure-ai-evaluation SDK</span> + "
    "<span class='custom-tag'>custom retrieval evaluator</span>.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Input ────────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-label">New Evaluation</div>', unsafe_allow_html=True)

question = st.text_area(
    "Question",
    placeholder="e.g. What is the ESCO record retention requirement for customer authorizations in New York?",
    height=90, key="q_input",
)
ground_truth = st.text_area(
    "Your Expected Answer  *(Ground Truth)*",
    placeholder="e.g. ESCOs must retain customer authorization records for a minimum of 2 years per the NY UBP…",
    height=120, key="gt_input",
)

run_col, _ = st.columns([2, 5])
with run_col:
    run_btn = st.button(
        "▶  Run Evaluation",
        type="primary",
        use_container_width=True,
        disabled=(not search_client or sdk_evaluators is None),
    )

# ── Execute ───────────────────────────────────────────────────────────────────

if run_btn:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not ground_truth.strip():
        st.warning("Please enter your expected answer.")
    else:
        status_box = st.empty()

        def update_status(msg):
            status_box.markdown(f'<div class="prog-pill">{msg}</div>', unsafe_allow_html=True)

        try:
            result = run_single_eval(
                question=question.strip(),
                ground_truth=ground_truth.strip(),
                search_client=search_client,
                openai_client=openai_client,
                embed_deploy=embed_deploy,
                chat_deploy=chat_deploy,
                sdk_evaluators=sdk_evaluators,
                top_k=top_k,
                delay=delay,
                status_fn=update_status,
            )
            st.session_state.results.insert(0, result)
            status_box.empty()
            st.rerun()
        except Exception as e:
            status_box.empty()
            st.error(f"Evaluation failed: {e}")


# ── Results ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<div class="sec-label">Evaluation Results</div>', unsafe_allow_html=True)

if not st.session_state.results:
    st.markdown("""
<div class="empty-state">
  <div class="es-icon">🧬</div>
  <div class="es-text">No evaluations yet — enter a question above and run.</div>
</div>""", unsafe_allow_html=True)

else:
    for idx, result in enumerate(st.session_state.results):
        ev   = result["eval"]
        comp = ev.get("composite_score")

        gd   = ev.get("groundedness",          {})
        rv   = ev.get("relevance",              {})
        rc   = ev.get("response_completeness",  {})
        rq   = ev.get("retrieval_quality",      {})

        q_num   = len(st.session_state.results) - idx
        q_short = result["question"][:72] + ("…" if len(result["question"]) > 72 else "")

        with st.expander(f"Q{q_num}  ·  {q_short}", expanded=(idx == 0)):

            # ── Score cards ────────────────────────────────────────────────
            c0, c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1, 1])
            with c0:
                st.markdown(render_composite(comp), unsafe_allow_html=True)
            with c1:
                st.markdown(render_dim_card(
                    "Groundedness", gd.get("score"), gd.get("source",""),
                    raw=gd.get("raw_score"), passed_str=gd.get("passed"), weight_pct=30
                ), unsafe_allow_html=True)
            with c2:
                st.markdown(render_dim_card(
                    "Relevance", rv.get("score"), rv.get("source",""),
                    raw=rv.get("raw_score"), passed_str=rv.get("passed"), weight_pct=25
                ), unsafe_allow_html=True)
            with c3:
                st.markdown(render_dim_card(
                    "Completeness", rc.get("score"), rc.get("source",""),
                    raw=rc.get("raw_score"), passed_str=rc.get("passed"), weight_pct=30
                ), unsafe_allow_html=True)
            with c4:
                st.markdown(render_dim_card(
                    "Retrieval", rq.get("score"), rq.get("source",""),
                    weight_pct=15
                ), unsafe_allow_html=True)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

            # ── Answer comparison ──────────────────────────────────────────
            st.markdown('<div class="sec-label">Answer Comparison</div>', unsafe_allow_html=True)
            left, right = st.columns(2)
            with left:
                st.markdown(f"""
<div class="answer-pane">
  <div class="pane-label gt">✅ Your Expected Answer</div>
  <div class="pane-text">{result["ground_truth_answer"]}</div>
</div>""", unsafe_allow_html=True)
            with right:
                st.markdown(f"""
<div class="answer-pane">
  <div class="pane-label gen">🤖 RAG Generated Answer</div>
  <div class="pane-text">{result["generated_answer"]}</div>
</div>""", unsafe_allow_html=True)

            if result.get("retrieved_sources"):
                chips = "".join(
                    f'<span class="src-chip">{s}</span>'
                    for s in result["retrieved_sources"] if s
                )
                st.markdown(
                    f'<div style="margin-top:10px">'
                    f'<span style="font-size:.65rem;color:#2d3748;text-transform:uppercase;'
                    f'letter-spacing:.1em;font-weight:700;">Retrieved from &nbsp;</span>'
                    f'{chips}</div>', unsafe_allow_html=True,
                )

            # ── Dimension detail tabs ──────────────────────────────────────
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Dimension Detail</div>', unsafe_allow_html=True)

            dt1, dt2, dt3, dt4 = st.tabs([
                "🧱 Groundedness (SDK)",
                "🎯 Relevance (SDK)",
                "✔️ Completeness (SDK)",
                "🔍 Retrieval (Custom)",
            ])

            with dt1:
                st.markdown(
                    f'<div class="detail-box">'
                    f'<strong>SDK Reason:</strong><br>{gd.get("reason","—")}<br><br>'
                    f'<strong>Raw score:</strong> {fmt_raw(gd.get("raw_score"))} &nbsp;·&nbsp; '
                    f'<strong>Normalised:</strong> {fmt(gd.get("score"))} &nbsp;·&nbsp; '
                    f'{passed_badge(gd.get("passed"))}'
                    f'</div>', unsafe_allow_html=True,
                )

            with dt2:
                st.markdown(
                    f'<div class="detail-box">'
                    f'<strong>SDK Reason:</strong><br>{rv.get("reason","—")}<br><br>'
                    f'<strong>Raw score:</strong> {fmt_raw(rv.get("raw_score"))} &nbsp;·&nbsp; '
                    f'<strong>Normalised:</strong> {fmt(rv.get("score"))} &nbsp;·&nbsp; '
                    f'{passed_badge(rv.get("passed"))}'
                    f'</div>', unsafe_allow_html=True,
                )

            with dt3:
                st.markdown(
                    f'<div class="detail-box">'
                    f'<strong>SDK Reason:</strong><br>{rc.get("reason","—")}<br><br>'
                    f'<strong>Raw score:</strong> {fmt_raw(rc.get("raw_score"))} &nbsp;·&nbsp; '
                    f'<strong>Normalised:</strong> {fmt(rc.get("score"))} &nbsp;·&nbsp; '
                    f'{passed_badge(rc.get("passed"))}'
                    f'</div>', unsafe_allow_html=True,
                )
                st.markdown("""
<div style="font-size:.72rem;color:#374151;margin-top:8px;padding:6px 10px;
            background:#0e1628;border-radius:6px;border:1px solid #1a2540;">
  💡 <strong style="color:#4a5568">ResponseCompletenessEvaluator</strong> measures whether
  the generated answer covers all critical information from your ground truth (recall-focused).
  Complements groundedness, which is precision-focused.
</div>""", unsafe_allow_html=True)

            with dt4:
                st.markdown(
                    f'<div class="detail-box">'
                    f'<strong>Reasoning:</strong><br>{rq.get("reasoning","—")}'
                    f'</div>', unsafe_allow_html=True,
                )
                cs = rq.get("chunk_scores", [])
                if cs:
                    st.markdown("**Chunk-level relevance scores:**")
                    pills_html = "".join(
                        f'<span class="pill {pill_cls(sc_val)}">Chunk {ci} → {fmt(sc_val)}</span>'
                        for ci, sc_val in enumerate(cs)
                    )
                    st.markdown(pills_html, unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                with m1:
                    top_rel = rq.get("top_chunk_is_relevant")
                    if top_rel is not None:
                        icon = "✅" if top_rel else "❌"
                        st.markdown(
                            f'<div class="detail-box">{icon} Top chunk relevant: <strong>{top_rel}</strong></div>',
                            unsafe_allow_html=True,
                        )
                with m2:
                    irr = rq.get("irrelevant_chunks_count")
                    if irr is not None:
                        st.markdown(
                            f'<div class="detail-box">🔢 Irrelevant chunks: <strong>{irr}</strong> / {result["num_chunks"]}</div>',
                            unsafe_allow_html=True,
                        )
                st.markdown("""
<div style="font-size:.72rem;color:#374151;margin-top:8px;padding:6px 10px;
            background:#0e1628;border-radius:6px;border:1px solid #1a2540;">
  💡 This dimension uses a <strong style="color:#4a5568">custom LLM-as-judge</strong>
  because the azure-ai-evaluation SDK's built-in Retrieval evaluator does not
  produce chunk-level scores — it only gives a single pass/fail for the full context.
</div>""", unsafe_allow_html=True)
