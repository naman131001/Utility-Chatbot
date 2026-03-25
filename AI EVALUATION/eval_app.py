"""
eval_app.py  —  RAG Evaluator Streamlit App
--------------------------------------------
Simple focused flow:
  1. User enters: Question + Their Expected Answer
  2. App runs RAG pipeline → gets Generated Answer
  3. LLM evaluates across 4 dimensions
  4. Shows score cards + expandable detail per row

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
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Evaluator",
    # page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #080c14; }
[data-testid="stSidebar"] {
    background: #0c1120;
    border-right: 1px solid #141d2e;
}

/* ── Score cards ── */
.dim-card {
    background: #0e1628;
    border: 1px solid #1a2540;
    border-radius: 12px;
    padding: 20px 16px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.dim-card .accent-bar {
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.dim-card .dim-score {
    font-family: 'DM Mono', monospace;
    font-size: 2.4rem; font-weight: 500; line-height: 1; margin: 6px 0 4px;
}
.dim-card .dim-name {
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #4a5568; margin-bottom: 6px;
}
.dim-card .dim-weight { font-size: 0.65rem; color: #2d3748; font-family: 'DM Mono', monospace; }

/* score colours */
.score-hi   { color: #34d399; }
.score-mid  { color: #fbbf24; }
.score-lo   { color: #f87171; }
.score-none { color: #374151; }

.bar-green { background: linear-gradient(90deg,#34d399,#059669); }
.bar-amber { background: linear-gradient(90deg,#fbbf24,#d97706); }
.bar-red   { background: linear-gradient(90deg,#f87171,#dc2626); }
.bar-gray  { background: #1f2d44; }
.bar-blue  { background: linear-gradient(90deg,#60a5fa,#3b82f6); }

/* ── Composite ── */
.composite-wrap {
    background: #0e1628; border: 1px solid #1a2540;
    border-radius: 12px; padding: 20px; text-align: center;
}
.composite-val {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem; font-weight: 800; line-height: 1;
}
.composite-label {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em;
    color: #4a5568; font-weight: 600; margin-top: 6px;
}

/* ── Answer panes ── */
.answer-pane {
    background: #080c14; border: 1px solid #141d2e;
    border-radius: 8px; padding: 14px 16px;
}
.pane-label {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em;
    font-weight: 700; margin-bottom: 10px; padding-bottom: 6px;
    border-bottom: 1px solid #141d2e;
}
.pane-label.gt  { color: #34d399; }
.pane-label.gen { color: #60a5fa; }
.pane-text { font-size: 0.84rem; color: #94a3b8; line-height: 1.6; white-space: pre-wrap; }

/* ── Detail / reasoning ── */
.detail-box {
    background: #080c14; border: 1px dashed #1a2540; border-radius: 8px;
    padding: 12px 16px; font-size: 0.82rem; color: #64748b; line-height: 1.6; margin-top: 8px;
}
.detail-box strong { color: #94a3b8; }

/* ── Issue / ok items ── */
.issue-item {
    background: #120a0a; border-left: 2px solid #f87171;
    border-radius: 0 6px 6px 0; padding: 6px 12px;
    font-size: 0.8rem; color: #fca5a5; margin: 4px 0;
}
.ok-item {
    background: #061412; border-left: 2px solid #34d399;
    border-radius: 0 6px 6px 0; padding: 6px 12px;
    font-size: 0.8rem; color: #6ee7b7; margin: 4px 0;
}

/* ── Pills ── */
.pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 20px;
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    font-weight: 500; margin: 0 4px 4px 0;
}
.pill-hi   { background:#064e3b; color:#34d399; border:1px solid #065f46; }
.pill-mid  { background:#451a03; color:#fbbf24; border:1px solid #78350f; }
.pill-lo   { background:#450a0a; color:#f87171; border:1px solid #7f1d1d; }
.pill-blue { background:#1e3a6e; color:#60a5fa; border:1px solid #1d4ed8; }

/* ── Source chips ── */
.src-chip {
    display: inline-block; background: #0e1628; border: 1px solid #1a2540;
    color: #64748b; border-radius: 4px; padding: 2px 8px;
    font-size: 0.68rem; font-family: 'DM Mono', monospace; margin: 2px 3px 2px 0;
}

/* ── Section label ── */
.sec-label {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.14em;
    color: #2d3748; font-weight: 700; margin: 20px 0 10px;
    padding-bottom: 6px; border-bottom: 1px solid #111827;
}

/* ── Progress ── */
.prog-pill {
    background: #0e1628; border: 1px solid #1a2540; border-radius: 8px;
    padding: 10px 16px; font-family: 'DM Mono', monospace;
    font-size: 0.78rem; color: #60a5fa; margin: 4px 0;
}

/* ── Empty state ── */
.empty-state {
    background: #0e1628; border: 1px dashed #1a2540;
    border-radius: 14px; padding: 48px 24px; text-align: center;
}
.empty-state .es-icon { font-size: 2.4rem; margin-bottom: 10px; }
.empty-state .es-text { font-family: 'Syne', sans-serif; font-size: 1rem; color: #374151; }

/* ── Global ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #f1f5f9 !important; }
p, li { color: #64748b; }
.stTextArea textarea, .stTextInput input {
    background: #080c14 !important; border-color: #1a2540 !important; color: #e2e8f0 !important;
}
.stButton > button { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Azure clients
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
    openai_client = AzureOpenAI(azure_endpoint=aoai_ep, api_key=aoai_key, api_version="2024-02-01")
    embed_deploy  = os.environ.get("AZURE_OPENAI_EMBED_DEPLOY", "text-embedding-3-small")
    chat_deploy   = os.environ.get("AZURE_OPENAI_CHAT_DEPLOY",  "gpt-4o-mini")
    return search_client, openai_client, embed_deploy, chat_deploy


# ─────────────────────────────────────────────────────────────────────────────
# RAG pipeline helpers
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
# Evaluators (LLM-as-judge)
# ─────────────────────────────────────────────────────────────────────────────

EVAL_SYSTEM = """You are a strict expert evaluator for a RAG system on NY retail energy regulatory documents.
Respond ONLY with a valid JSON object. No preamble, no markdown fences."""

def _llm_eval(prompt, openai_client, chat_deploy):
    try:
        raw = openai_client.chat.completions.create(
            model=chat_deploy,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0, max_tokens=500,
        ).choices[0].message.content.strip()
        raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e), "score": 0.0}


def eval_groundedness(question, answer, hits, client, deploy):
    ctx = build_context(hits[:6])
    r = _llm_eval(f"""Evaluate GROUNDEDNESS of this answer.
Every factual claim must be supported by the source chunks. Penalise unsupported claims.

Question: {question}
Source Chunks:\n{ctx}
Generated Answer:\n{answer}

Return JSON:
{{"score": <0.0-1.0>, "reasoning": "<one paragraph>", "unsupported_claims": ["claim1"]}}

1.0=fully grounded, 0.7=minor extrapolation, 0.4=some unsupported, 0.0=contradicts sources""",
        client, deploy)
    r["dimension"] = "groundedness"
    return r


def eval_relevance(question, answer, client, deploy):
    r = _llm_eval(f"""Evaluate RELEVANCE — does the answer address what was actually asked?

Question: {question}
Generated Answer:\n{answer}

Return JSON:
{{"score": <0.0-1.0>, "reasoning": "<one paragraph>", "missing_aspects": ["aspect1"]}}

1.0=fully on-topic, 0.7=minor tangents, 0.4=key aspects ignored, 0.0=off-topic""",
        client, deploy)
    r["dimension"] = "relevance"
    return r


def eval_correctness(question, generated, ground_truth, client, deploy):
    r = _llm_eval(f"""Evaluate CORRECTNESS by comparing generated answer to ground truth.
Paraphrasing is fine — key facts, rules, and figures must match.

Question: {question}
Ground Truth Answer:\n{ground_truth}
Generated Answer:\n{generated}

Return JSON:
{{"score": <0.0-1.0>, "reasoning": "<one paragraph>",
  "correct_facts": ["fact1"], "incorrect_or_missing_facts": ["fact2"]}}

1.0=all facts match, 0.7=mostly correct, 0.4=significant gaps, 0.0=contradicts truth""",
        client, deploy)
    r["dimension"] = "correctness"
    return r


def eval_retrieval(question, hits, client, deploy):
    chunks_text = "".join(
        f"\n[CHUNK {i}] {h.get('source_pdf_name','?')}\n{h.get('content','')[:300]}\n"
        for i, h in enumerate(hits[:8])
    )
    r = _llm_eval(f"""Evaluate RETRIEVAL QUALITY. Score each retrieved chunk's relevance (0-1)
and give an overall retrieval score.

Question: {question}
Retrieved Chunks:\n{chunks_text}

Return JSON:
{{"overall_score": <0.0-1.0>, "reasoning": "<one paragraph>",
  "chunk_scores": [0.9, 0.4],
  "top_chunk_is_relevant": true,
  "irrelevant_chunks_count": 2}}

1.0=all relevant, 0.7=mostly relevant, 0.4=mixed, 0.0=entirely off""",
        client, deploy)
    r["dimension"] = "retrieval_quality"
    return r


WEIGHTS = {"groundedness": 0.30, "relevance": 0.25, "correctness": 0.30, "retrieval_quality": 0.15}

def composite_score(ev):
    ws = wt = 0.0
    for dim, w in WEIGHTS.items():
        s = ev.get(dim, {}).get("score") or ev.get(dim, {}).get("overall_score")
        if s is not None:
            ws += s * w; wt += w
    return round(ws / wt, 4) if wt else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_single_eval(question, ground_truth, search_client, openai_client,
                    embed_deploy, chat_deploy, top_k, delay, status_fn=None):

    def step(msg):
        if status_fn: status_fn(msg)

    step("🔍 Retrieving chunks from Azure AI Search…")
    hits = hybrid_search(question, search_client, openai_client, embed_deploy, top_k)
    time.sleep(delay)

    step(f"🤖 Generating answer ({len(hits)} chunks retrieved)…")
    generated = generate_answer(question, hits, openai_client, chat_deploy)
    time.sleep(delay)

    step("📐 Evaluating groundedness…")
    g = eval_groundedness(question, generated, hits, openai_client, chat_deploy)
    time.sleep(delay)

    step("🎯 Evaluating relevance…")
    r = eval_relevance(question, generated, openai_client, chat_deploy)
    time.sleep(delay)

    step("✔️ Evaluating correctness vs your answer…")
    c = eval_correctness(question, generated, ground_truth, openai_client, chat_deploy)
    time.sleep(delay)

    step("🔎 Evaluating retrieval quality…")
    rq = eval_retrieval(question, hits, openai_client, chat_deploy)
    time.sleep(delay)

    ev = {"groundedness": g, "relevance": r, "correctness": c, "retrieval_quality": rq}
    ev["composite_score"] = composite_score(ev)

    return {
        "question":            question,
        "ground_truth_answer": ground_truth,
        "generated_answer":    generated,
        "retrieved_sources":   [h.get("source_pdf_name", h.get("source", "")) for h in hits],
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

def render_dim_card(label, score, weight_pct):
    sc, bc = score_cls(score)
    return f"""
<div class="dim-card">
  <div class="accent-bar {bc}"></div>
  <div class="dim-name">{label}</div>
  <div class="dim-score {sc}">{fmt(score)}</div>
  <div class="dim-weight">weight {weight_pct}%</div>
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
    st.markdown("## RAG Evaluator")
    st.markdown("---")

    search_client, openai_client, embed_deploy, chat_deploy = get_clients()

    if not search_client:
        st.error("⚠️ Azure credentials missing. Check your .env file.")
    else:
        st.success("✅ Azure connected")

    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", 5, 20, 15)
    delay = st.slider("API delay (s)", 0.5, 3.0, 1.0, 0.5,
                      help="Pause between LLM calls to avoid rate limits")

    st.markdown("---")
    st.markdown("### 📐 Dimension Weights")
    dim_meta = [
        ("Groundedness",      "#34d399", "30%"),
        ("Relevance",         "#a78bfa", "25%"),
        ("Correctness",       "#fbbf24", "30%"),
        ("Retrieval Quality", "#60a5fa", "15%"),
    ]
    for label, color, pct in dim_meta:
        st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            margin-bottom:6px;padding:6px 10px;background:#0e1628;
            border-radius:6px;border:1px solid #1a2540;">
  <span style="font-size:0.78rem;color:#64748b;">{label}</span>
  <span style="font-family:'DM Mono',monospace;font-size:0.78rem;color:{color};">{pct}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Stats
    n_total = len(st.session_state.results)
    scores  = [r["eval"].get("composite_score") for r in st.session_state.results
               if r["eval"].get("composite_score") is not None]
    avg     = round(sum(scores) / len(scores), 2) if scores else None
    sc_cls, _ = score_cls(avg)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
<div style="background:#0e1628;border:1px solid #1a2540;border-radius:8px;
            padding:12px;text-align:center;">
  <div style="font-family:'DM Mono',monospace;font-size:1.6rem;color:#60a5fa;">{n_total}</div>
  <div style="font-size:0.65rem;color:#2d3748;text-transform:uppercase;letter-spacing:0.1em;">Evaluated</div>
</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
<div style="background:#0e1628;border:1px solid #1a2540;border-radius:8px;
            padding:12px;text-align:center;">
  <div style="font-family:'DM Mono',monospace;font-size:1.6rem;" class="{sc_cls}">{fmt(avg)}</div>
  <div style="font-size:0.65rem;color:#2d3748;text-transform:uppercase;letter-spacing:0.1em;">Avg Score</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear all results", use_container_width=True):
        st.session_state.results = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# RAG Evaluator")
st.markdown(
    "<p style='color:#374151;margin-top:-8px;'>Enter a question and your expected answer — "
    "the app retrieves, generates, then scores the pipeline response.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-label">New Evaluation</div>', unsafe_allow_html=True)

question = st.text_area(
    "Question",
    placeholder="e.g. What is the ESCO record retention requirement for customer authorizations in New York?",
    height=90,
    key="q_input",
)

ground_truth = st.text_area(
    "Your Expected Answer  *(Ground Truth)*",
    placeholder="e.g. ESCOs must retain customer authorization records for a minimum of 2 years per the NY UBP…",
    height=120,
    key="gt_input",
)

run_col, _ = st.columns([2, 5])
with run_col:
    run_btn = st.button(
        "▶  Run Evaluation",
        type="primary",
        use_container_width=True,
        disabled=(not search_client),
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
            status_box.markdown(
                f'<div class="prog-pill">{msg}</div>', unsafe_allow_html=True
            )

        try:
            result = run_single_eval(
                question=question.strip(),
                ground_truth=ground_truth.strip(),
                search_client=search_client,
                openai_client=openai_client,
                embed_deploy=embed_deploy,
                chat_deploy=chat_deploy,
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
        g    = ev.get("groundedness",     {}).get("score")
        r    = ev.get("relevance",        {}).get("score")
        c    = ev.get("correctness",      {}).get("score")
        rq   = ev.get("retrieval_quality",{}).get("overall_score")

        q_num   = len(st.session_state.results) - idx
        q_short = result["question"][:72] + ("…" if len(result["question"]) > 72 else "")

        with st.expander(f"Q{q_num}  ·  {q_short}", expanded=(idx == 0)):

            # ── Score cards ────────────────────────────────────────────────
            c0, c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1, 1])
            with c0: st.markdown(render_composite(comp), unsafe_allow_html=True)
            with c1: st.markdown(render_dim_card("Groundedness", g, 30), unsafe_allow_html=True)
            with c2: st.markdown(render_dim_card("Relevance", r, 25), unsafe_allow_html=True)
            with c3: st.markdown(render_dim_card("Correctness", c, 30), unsafe_allow_html=True)
            with c4: st.markdown(render_dim_card("Retrieval", rq, 15), unsafe_allow_html=True)

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

            # Retrieved sources row
            if result.get("retrieved_sources"):
                chips = "".join(
                    f'<span class="src-chip">{s}</span>'
                    for s in result["retrieved_sources"] if s
                )
                st.markdown(
                    f'<div style="margin-top:10px">'
                    f'<span style="font-size:0.65rem;color:#2d3748;text-transform:uppercase;'
                    f'letter-spacing:0.1em;font-weight:700;">Retrieved from &nbsp;</span>'
                    f'{chips}</div>',
                    unsafe_allow_html=True,
                )

            # ── Dimension detail tabs ──────────────────────────────────────
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Dimension Detail</div>', unsafe_allow_html=True)

            dt1, dt2, dt3, dt4 = st.tabs([
                "🧱 Groundedness", "🎯 Relevance", "✔️ Correctness", "🔍 Retrieval"
            ])

            with dt1:
                gd = ev.get("groundedness", {})
                st.markdown(
                    f'<div class="detail-box"><strong>Reasoning:</strong><br>{gd.get("reasoning","—")}</div>',
                    unsafe_allow_html=True,
                )
                unsup = gd.get("unsupported_claims", [])
                if unsup:
                    st.markdown("**⚠️ Unsupported claims:**")
                    st.markdown(render_issues(unsup, "issue"), unsafe_allow_html=True)
                else:
                    st.markdown(
                        render_issues(["All claims grounded in source chunks"], "ok"),
                        unsafe_allow_html=True,
                    )

            with dt2:
                rv = ev.get("relevance", {})
                st.markdown(
                    f'<div class="detail-box"><strong>Reasoning:</strong><br>{rv.get("reasoning","—")}</div>',
                    unsafe_allow_html=True,
                )
                miss = rv.get("missing_aspects", [])
                if miss:
                    st.markdown("**⚠️ Aspects not addressed:**")
                    st.markdown(render_issues(miss, "issue"), unsafe_allow_html=True)
                else:
                    st.markdown(
                        render_issues(["Answer fully addresses the question"], "ok"),
                        unsafe_allow_html=True,
                    )

            with dt3:
                cv = ev.get("correctness", {})
                st.markdown(
                    f'<div class="detail-box"><strong>Reasoning:</strong><br>{cv.get("reasoning","—")}</div>',
                    unsafe_allow_html=True,
                )
                cf1, cf2 = st.columns(2)
                with cf1:
                    correct = cv.get("correct_facts", [])
                    if correct:
                        st.markdown("**✅ Correct facts:**")
                        st.markdown(render_issues(correct, "ok"), unsafe_allow_html=True)
                with cf2:
                    wrong = cv.get("incorrect_or_missing_facts", [])
                    if wrong:
                        st.markdown("**❌ Incorrect / missing:**")
                        st.markdown(render_issues(wrong, "issue"), unsafe_allow_html=True)

            with dt4:
                rqv = ev.get("retrieval_quality", {})
                st.markdown(
                    f'<div class="detail-box"><strong>Reasoning:</strong><br>{rqv.get("reasoning","—")}</div>',
                    unsafe_allow_html=True,
                )
                cs = rqv.get("chunk_scores", [])
                if cs:
                    st.markdown("**Chunk-level relevance scores:**")
                    pills_html = "".join(
                        f'<span class="pill {pill_cls(sc_val)}">Chunk {ci} → {fmt(sc_val)}</span>'
                        for ci, sc_val in enumerate(cs)
                    )
                    st.markdown(pills_html, unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                with m1:
                    top_rel = rqv.get("top_chunk_is_relevant")
                    if top_rel is not None:
                        icon = "✅" if top_rel else "❌"
                        st.markdown(
                            f'<div class="detail-box">{icon} Top chunk relevant: <strong>{top_rel}</strong></div>',
                            unsafe_allow_html=True,
                        )
                with m2:
                    irr = rqv.get("irrelevant_chunks_count")
                    if irr is not None:
                        st.markdown(
                            f'<div class="detail-box">🔢 Irrelevant chunks: <strong>{irr}</strong> / {result["num_chunks"]}</div>',
                            unsafe_allow_html=True,
                        )
