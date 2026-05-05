"""
ingest.py — PDF parsing, chunking, and vector store population.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_voyageai import VoyageAIEmbeddings

COLLECTION_NAME = "fs_rulebook"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_MODEL = "voyage-3"


def extract_pages(pdf_path: Path) -> List[dict]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append({"page": i + 1, "text": text, "source": pdf_path.name})
    doc.close()
    return pages


def chunk_pages(pages: List[dict]) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        for split in splitter.split_text(page["text"]):
            chunks.append({"text": split, "metadata": {"page": page["page"], "source": page["source"]}})
    return chunks


def ingest_pdf(pdf_paths: List[Path], persist_dir: str = "./chroma_store") -> Chroma:
    all_chunks: List[dict] = []
    for path in pdf_paths:
        pages = extract_pages(path)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    voyage_key = os.environ.get("VOYAGE_API_KEY", "")
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=voyage_key,
        model=EMBED_MODEL,
    )

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    return vectorstore


if __name__ == "__main__":
    import sys
    paths = [Path(p) for p in sys.argv[1:]]
    ingest_pdf(paths)
