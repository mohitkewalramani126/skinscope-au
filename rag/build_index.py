"""
One-time (re-runnable) script: embeds every chunk in rag/corpus.json and
writes rag/embeddings.json, a small precomputed-embeddings file used by
rag/retriever.py for plain numpy cosine-similarity search. Re-run this any
time corpus.json changes.

Day 14 rewrite: this used to build a Chroma persistent vector DB via
sentence-transformers + chromadb. Both were dropped from the runtime (see
rag/retriever.py's module docstring) to fit a free-tier serverless host's
memory/bundle-size limits. For a 20-chunk corpus, a full vector database was
always somewhat oversized anyway -- a flat numpy array and a linear scan
over 20 rows costs microseconds, so this isn't a quality tradeoff, just a
right-sized replacement.

rag/embeddings.json is small (~30KB for 20 chunks x 384 dims) and is
committed to the repo, unlike the old chroma_db/ directory -- there's
nothing to gitignore or rebuild at container start anymore.

Requires rag/embedding_model/ to exist first -- run
rag/download_embedding_model.py once if it doesn't.
"""

import json
import os

from rag.embedder import embed

RAG_DIR = os.path.dirname(__file__)
CORPUS_PATH = os.path.join(RAG_DIR, "corpus.json")
EMBEDDINGS_PATH = os.path.join(RAG_DIR, "embeddings.json")


def build_index():
    with open(CORPUS_PATH, "r") as f:
        corpus = json.load(f)

    print(f"Loaded {len(corpus)} chunks from {CORPUS_PATH}")

    texts = [chunk["text"] for chunk in corpus]
    embeddings = embed(texts)

    records = [
        {
            "id": chunk["id"],
            "text": chunk["text"],
            "topic": chunk["topic"],
            "source_title": chunk["source_title"],
            "source_url": chunk["source_url"],
            "embedding": embeddings[i].tolist(),
        }
        for i, chunk in enumerate(corpus)
    ]

    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(records, f)

    print(f"Wrote {len(records)} embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    build_index()
