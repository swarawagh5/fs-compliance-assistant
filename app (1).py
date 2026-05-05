"""
Formula Student Technical Compliance Assistant — Streamlit UI
"""

import streamlit as st
import os
from pathlib import Path

from ingest import ingest_pdf
from retrieval import ComplianceRetriever

st.set_page_config(
    page_title="FS Compliance Assistant",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Rajdhani', sans-serif; }

.stApp { background: #0a0c10; color: #e2e8f0; }

.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #e53e3e 0%, #fc8181 50%, #fbd38d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
}

.rule-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #e53e3e;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #cdd6f4;
}

.answer-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.4rem;
    margin-top: 1rem;
    color: #e2e8f0;
    line-height: 1.7;
}

.badge {
    display: inline-block;
    background: #e53e3e22;
    color: #fc8181;
    border: 1px solid #e53e3e55;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-right: 4px;
}

.score-pill {
    background: #1a472a;
    color: #68d391;
    border: 1px solid #2f855a;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
}

div[data-testid="stFileUploader"] { border: 1px dashed #30363d; border-radius: 8px; padding: 0.5rem; }
.stTextInput > div > input { background: #161b22; color: #e2e8f0; border: 1px solid #30363d; border-radius: 6px; }
.stButton > button { background: #e53e3e; color: white; border: none; border-radius: 6px; font-family: 'Rajdhani', sans-serif; font-weight: 600; font-size: 1rem; letter-spacing: 0.5px; transition: all 0.2s; }
.stButton > button:hover { background: #c53030; transform: translateY(-1px); }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hero-title">🏎️ FS Compliance</p>', unsafe_allow_html=True)
    st.caption("Formula Student Technical Compliance Assistant")
    st.divider()

    st.markdown("**🔑 API Keys**")
    anthropic_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    voyage_key = st.text_input("Voyage API Key", type="password", placeholder="pa-...",
                               help="Get a free key at dash.voyageai.com")

    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if voyage_key:
        os.environ["VOYAGE_API_KEY"] = voyage_key

    # Load keys from Streamlit secrets if available (for cloud deployment)
    if not anthropic_key and "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
        anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
    if not voyage_key and "VOYAGE_API_KEY" in st.secrets:
        os.environ["VOYAGE_API_KEY"] = st.secrets["VOYAGE_API_KEY"]
        voyage_key = st.secrets["VOYAGE_API_KEY"]

    st.divider()
    st.markdown("**📁 Document Ingestion**")

    uploaded_files = st.file_uploader(
        "Upload Rulebook / Design Docs (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    persist_dir = Path("./chroma_store")

    if st.button("⚡ Ingest & Index Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        elif not anthropic_key:
            st.warning("Please enter your Anthropic API key.")
        elif not voyage_key:
            st.warning("Please enter your Voyage API key.")
        else:
            with st.spinner("Parsing, chunking, and embedding…"):
                tmp_paths = []
                for f in uploaded_files:
                    tmp_path = Path(f"/tmp/{f.name}")
                    tmp_path.write_bytes(f.read())
                    tmp_paths.append(tmp_path)
                try:
                    ingest_pdf(tmp_paths, persist_dir=str(persist_dir))
                    st.session_state.retriever = ComplianceRetriever(
                        persist_dir=str(persist_dir),
                        api_key=anthropic_key,
                        voyage_key=voyage_key,
                    )
                    st.session_state.docs_loaded = [f.name for f in uploaded_files]
                    st.success(f"✅ Indexed {len(uploaded_files)} document(s)")
                except Exception as e:
                    st.error(f"Ingestion error: {e}")

    if persist_dir.exists() and st.session_state.retriever is None and anthropic_key and voyage_key:
        if st.button("📂 Load Existing Index", use_container_width=True):
            try:
                st.session_state.retriever = ComplianceRetriever(
                    persist_dir=str(persist_dir),
                    api_key=anthropic_key,
                    voyage_key=voyage_key,
                )
                st.success("Existing vector store loaded.")
            except Exception as e:
                st.error(f"Load error: {e}")

    if st.session_state.docs_loaded:
        st.divider()
        st.markdown("**📄 Indexed Documents**")
        for doc in st.session_state.docs_loaded:
            st.markdown(f'<span class="badge">PDF</span> {doc}', unsafe_allow_html=True)

    st.divider()
    st.caption("Built with ChromaDB · LangChain · Claude Sonnet 4")

# ── Main panel ────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<h1 class="hero-title">Technical Compliance Assistant</h1>', unsafe_allow_html=True)
    st.markdown("*Ask compliance questions — get rule citations with AI analysis*")
with col_status:
    if st.session_state.retriever:
        st.success("● Index ready")
    else:
        st.warning("● No index loaded")

st.divider()

st.markdown("**⚡ Example queries:**")
examples = [
    "Does our main hoop height meet rule T3.3?",
    "What are the nose cone impact attenuator requirements?",
    "Summarise all roll hoop bracing angle constraints.",
    "What fire suppression rules apply to EV vehicles?",
]
cols = st.columns(len(examples))
for i, (col, ex) in enumerate(zip(cols, examples)):
    with col:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["prefill"] = ex

prefill = st.session_state.pop("prefill", "")
query = st.text_input(
    "Compliance Question",
    value=prefill,
    placeholder="e.g. Does our main hoop height of 950mm meet rule T3.3?",
    label_visibility="collapsed",
)

col_ask, col_clear = st.columns([5, 1])
with col_ask:
    ask = st.button("🔍 Query Rulebook", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

if ask and query:
    if not st.session_state.retriever:
        st.error("Please ingest documents first (sidebar).")
    else:
        with st.spinner("Retrieving relevant rules and generating compliance analysis…"):
            try:
                result = st.session_state.retriever.query(query)
                st.session_state.chat_history.insert(0, {"query": query, "result": result})
            except Exception as e:
                st.error(f"Query error: {e}")

for entry in st.session_state.chat_history:
    q = entry["query"]
    r = entry["result"]

    st.markdown(f"### ❓ {q}")

    with st.expander(f"📋 Retrieved Rule Excerpts ({len(r['sources'])} passages)", expanded=True):
        for i, src in enumerate(r["sources"]):
            score_html = f'<span class="score-pill">Relevance: {src["score"]:.2f}</span>' if "score" in src else ""
            st.markdown(
                f'<span class="badge">CHUNK {i+1}</span> '
                f'<span class="badge">{src.get("source","PDF")}</span> '
                f'Page {src.get("page","?")} &nbsp; {score_html}',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="rule-card">{src["text"]}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="answer-card">{r["answer"]}</div>', unsafe_allow_html=True)
    st.divider()
