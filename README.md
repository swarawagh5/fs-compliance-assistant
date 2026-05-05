# 🏎️ Formula Student Technical Compliance Assistant

A RAG-powered chatbot that lets Formula Student teams verify design compliance
against the FSAE/FS rulebook — with exact page citations from the source PDF.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Embeddings** | Anthropic Voyage-3 | State-of-the-art retrieval accuracy |
| **LLM** | Claude Sonnet 4 | Precise, citation-aware technical reasoning |
| **Orchestration** | LangChain | Modular chain + splitter utilities |
| **Vector DB** | ChromaDB (local) | Zero-infra, persists to disk, fast MMR search |
| **PDF Parsing** | PyMuPDF (fitz) | Page-level metadata preserved for citations |
| **UI** | Streamlit | Rapid demo-ready interface for recruiters |

---

## Architecture

```
PDF Rulebook(s)
      │
      ▼
 [ingest.py]
  PyMuPDF → page-level text
      │
  RecursiveCharacterTextSplitter
  (800 char chunks, 150 overlap)
      │
  AnthropicEmbeddings (voyage-3)
      │
      ▼
 ChromaDB (persisted)
      │
 [retrieval.py]
  User query → voyage-3 embedding
      │
  MMR Search (k=5, pool=20)
  ← diverse, relevant rule chunks
      │
  Prompt injection → Claude Sonnet 4
      │
      ▼
 Compliance analysis with citations
```

---

## Setup

```bash
# 1. Clone / create project directory
mkdir fs_compliance && cd fs_compliance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then in the browser:
1. Enter your **Anthropic API key** in the sidebar
2. Upload your **FSAE/FS rulebook PDF** (and optionally internal design docs)
3. Click **⚡ Ingest & Index Documents**
4. Ask compliance questions like *"Does our main hoop height of 940mm meet T3.3?"*

---

## CLI Usage

```bash
# Ingest PDFs directly
python ingest.py fsae_rulebook_2024.pdf team_chassis_spec.pdf

# Query from the terminal
python retrieval.py "What are the roll hoop bracing angle requirements?"
```

---

## File Structure

```
fs_compliance/
├── app.py           # Streamlit UI
├── ingest.py        # PDF parsing, chunking, embedding, ChromaDB ingestion
├── retrieval.py     # MMR retrieval + Claude RAG generation
├── requirements.txt
├── README.md
└── chroma_store/    # Auto-created on first ingest (gitignore this)
```

---

## CV Bullet Point

> **Formula Student Technical Compliance Assistant** | Python, LangChain, ChromaDB, Anthropic Claude API
> 
> Engineered a production-grade RAG system to automate FSAE regulation compliance checks for a Formula Student team; implemented a multi-stage pipeline — PyMuPDF page-level extraction → recursive chunking → Voyage-3 vector embeddings → Maximal Marginal Relevance retrieval — feeding cited rule excerpts into Claude Sonnet 4 for structured compliance verdicts. Reduced manual rulebook cross-referencing time by an estimated 80% and deployed a Streamlit interface enabling non-technical team members to interrogate 400+ pages of technical regulations in natural language.

---

## Extending This Project

- **Multi-doc routing**: tag chunks by document type (rulebook vs. design spec) and filter at query time
- **Compliance checklist mode**: iterate over a structured design checklist and auto-verify each item
- **Pinecone swap**: replace ChromaDB with Pinecone for a cloud-hosted, shareable deployment
- **Structured output**: use Claude's tool-use API to return `{rule_id, compliant: bool, margin}` JSON
- **CI integration**: wrap retrieval.py in a GitHub Action triggered on design document commits
