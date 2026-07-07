"""
One-time (re-runnable) script: embeds every chunk in rag/corpus.json with a
sentence-transformers model and stores them in a local persistent Chroma
collection. Re-run this any time corpus.json changes.

Chunking strategy: each corpus.json entry is already a single, topic-scoped
paragraph (one idea: ABCDE, one skin-cancer type, one prevention step, etc.),
sized to a few sentences. This is a deliberate choice over splitting a big
document into arbitrary fixed-length windows — every chunk here is already
a coherent, independently-citable unit, so no further splitting is needed.
"""

import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

RAG_DIR = os.path.dirname(__file__)
CORPUS_PATH = os.path.join(RAG_DIR, "corpus.json")
CHROMA_DB_PATH = os.path.join(RAG_DIR, "chroma_db")
COLLECTION_NAME = "skinscope_corpus"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def build_index():
    with open(CORPUS_PATH, "r") as f:
        corpus = json.load(f)

    print(f"Loaded {len(corpus)} chunks from {CORPUS_PATH}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [chunk["text"] for chunk in corpus]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # drop any stale collection so re-running this script is always safe
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[chunk["id"] for chunk in corpus],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "topic": chunk["topic"],
                "source_title": chunk["source_title"],
                "source_url": chunk["source_url"],
            }
            for chunk in corpus
        ],
    )

    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}' at {CHROMA_DB_PATH}")


if __name__ == "__main__":
    build_index()