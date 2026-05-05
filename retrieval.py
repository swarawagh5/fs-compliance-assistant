"""
retrieval.py — Vector retrieval + Claude RAG answer generation.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic
from langchain_community.vectorstores import Chroma
from langchain_voyageai import VoyageAIEmbeddings

COLLECTION_NAME = "fs_rulebook"
EMBED_MODEL = "voyage-3"
TOP_K = 5
FETCH_K = 20

SYSTEM_PROMPT = """You are a senior Formula Student Technical Officer with deep expertise in
FSAE / Formula Student technical regulations. Your role is to assist engineering teams in
verifying design compliance against the official rulebook.

Guidelines:
- Be precise and technical. Reference rule numbers explicitly when they appear in the provided excerpts.
- Structure your answer: (1) Relevant Rules Found, (2) Compliance Analysis, (3) Verdict.
- If the retrieved excerpts do not contain enough information to make a definitive ruling, say so clearly.
- Use metric units consistently. Flag any ambiguity in the question.
- Be conservative: if compliance is uncertain, flag it for scrutineering review.
- Do NOT hallucinate rule numbers or thresholds not present in the provided text."""


def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    excerpt_block = ""
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        excerpt_block += f"\n--- EXCERPT {i} | Source: {source} | Page {page} ---\n{chunk['text']}\n"

    return f"""RETRIEVED RULE EXCERPTS:
{excerpt_block}

COMPLIANCE QUESTION:
{query}

Using ONLY the rule excerpts provided above, perform a compliance analysis.
Cite each excerpt by its number (e.g., "per Excerpt 2, Page 14…").
If you cannot find the specific rule, explicitly state which information is missing."""


class ComplianceRetriever:
    def __init__(self, persist_dir: str = "./chroma_store", api_key: str | None = None, voyage_key: str | None = None):
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        if voyage_key:
            os.environ["VOYAGE_API_KEY"] = voyage_key

        self.embeddings = VoyageAIEmbeddings(
            voyage_api_key=voyage_key or os.environ.get("VOYAGE_API_KEY", ""),
            model=EMBED_MODEL,
        )
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        results = self.vectorstore.max_marginal_relevance_search_with_score(
            query, k=TOP_K, fetch_k=FETCH_K,
        )
        chunks = []
        for doc, score in results:
            chunks.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": round(float(score), 4),
                "source": doc.metadata.get("source", "PDF"),
                "page": doc.metadata.get("page", "?"),
            })
        return chunks

    def generate(self, query: str, chunks: list[dict]) -> str:
        user_prompt = build_rag_prompt(query, chunks)
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    def query(self, question: str) -> dict[str, Any]:
        chunks = self.retrieve(question)
        if not chunks:
            return {
                "answer": "⚠️ No relevant rules found. Please ensure the correct rulebook has been ingested.",
                "sources": [],
            }
        answer = self.generate(question, chunks)
        return {"answer": answer, "sources": chunks}
