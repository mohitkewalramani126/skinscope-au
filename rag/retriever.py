"""
Retriever for SkinScope AU's RAG corpus.

Retrieves the top-k most relevant chunks from rag/embeddings.json (built by
build_index.py), using the same embedding model (via rag/embedder.py) so
query and corpus embeddings live in the same space.

Also enforces an out-of-scope gate: if the best match isn't similar enough to
the query, the corpus has nothing relevant to say, and the caller should refuse
rather than force an answer out of a weak match.

Day 14 rewrite: originally used chromadb + sentence-transformers (which pulls
in a full PyTorch install). Both were dropped in favor of rag/embedder.py
(onnxruntime + tokenizers, no torch) and a plain numpy cosine-similarity scan
over the 20-row corpus, to fit within a free-tier serverless host's bundle
size and memory limits. For a corpus this small, a full vector database
added dependency weight without adding capability -- 20 dot products is not
where a vector DB's indexing advantages matter.

This preserves the exact same public contract (RetrievedChunk, retrieve(),
is_in_scope(), OUT_OF_SCOPE_MESSAGE, answer_query()) as the Day 10 version,
so agent/nodes.py and every existing test needed no changes. The cosine
*distance* definition (1 - cosine_similarity) is also unchanged, so the
threshold below is the same number tuned in Day 10/11 -- re-verified after
this rewrite against evaluation/golden_qa.json rather than assumed to still
hold (see evaluation/rag_eval_report.md for the re-verification run).

Threshold tuning (evaluation/rag_eval.py, swept against evaluation/golden_qa.json,
27 questions / 22 in-scope, 5 out-of-scope):

  threshold  scope_accuracy
  0.50-0.65  92.59%  (25/27 -- widest band at the best accuracy)
  0.70-0.75  88.89%
  0.78-0.85  92.59%

0.65 is used: sits in the middle of the widest contiguous band (0.50-0.65)
that ties for the best accuracy, a more stable choice than a band edge.

The 2 remaining failures at 0.65 are both extremely short, context-free
questions ("What does ABCDE mean?", "Am I at risk?") -- a known, documented
blind spot (see evaluation/rag_eval_report.md), not something this threshold
choice can fix without conversational context this single-turn retriever
doesn't have. Re-verified numerically unchanged after the Day 14 embedder
rewrite (onnxruntime instead of sentence-transformers) -- same two failures,
same distances, confirming the ONNX reimplementation is behaviorally
faithful to the original model, not a regression.
"""

import json
import os
from dataclasses import dataclass

import numpy as np

from rag.embedder import embed

RAG_DIR = os.path.dirname(__file__)
EMBEDDINGS_PATH = os.path.join(RAG_DIR, "embeddings.json")

# Cosine "distance" is (1 - cosine_similarity), so smaller = more similar.
# Value chosen empirically — see module docstring for the threshold sweep.
OUT_OF_SCOPE_DISTANCE_THRESHOLD = 0.65

_corpus_records: list[dict] | None = None
_corpus_embeddings: np.ndarray | None = None  # (n_chunks, 384), L2-normalized


def _load_corpus():
    global _corpus_records, _corpus_embeddings
    if _corpus_records is None:
        with open(EMBEDDINGS_PATH, "r") as f:
            records = json.load(f)
        _corpus_records = records
        _corpus_embeddings = np.array([r["embedding"] for r in records], dtype=np.float32)
    return _corpus_records, _corpus_embeddings


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source_title: str
    source_url: str
    distance: float


def retrieve(query: str, k: int = 3) -> list[RetrievedChunk]:
    """Return the top-k chunks for `query`, ranked by cosine distance (ascending)."""
    records, corpus_embeddings = _load_corpus()

    query_embedding = embed([query])[0]  # already L2-normalized, shape (384,)
    # both sides are L2-normalized, so the dot product IS the cosine similarity
    similarities = corpus_embeddings @ query_embedding
    distances = 1.0 - similarities

    top_k_idx = np.argsort(distances)[:k]

    chunks = []
    for i in top_k_idx:
        record = records[i]
        chunks.append(RetrievedChunk(
            id=record["id"],
            text=record["text"],
            source_title=record["source_title"],
            source_url=record["source_url"],
            distance=float(distances[i]),
        ))
    return chunks


def is_in_scope(chunks: list[RetrievedChunk]) -> bool:
    """True if the best retrieved chunk is close enough to trust as relevant."""
    if not chunks:
        return False
    return chunks[0].distance <= OUT_OF_SCOPE_DISTANCE_THRESHOLD


OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions about skin cancer awareness, detection, prevention, "
    "and screening guidance, based on the cited sources in my corpus. I don't have "
    "relevant information to answer that. Please ask something related to skin "
    "checks, sun protection, or skin cancer types, or speak to a doctor for anything "
    "outside that."
)


@dataclass
class GroundedAnswer:
    answer: str
    citations: list[dict]  # [{"title": ..., "url": ...}, ...]
    refused: bool


def answer_query(query: str, k: int = 3) -> GroundedAnswer:
    """
    Single entry point enforcing grounding + out-of-scope refusal.

    Day 10: the "answer" is extractive (retrieved chunk text, concatenated,
    verbatim) — there is no LLM here yet. Day 11 wraps this exact contract
    (same citations list, same refusal gate) with Groq to phrase the answer in
    natural language, without changing what counts as grounded or in-scope.
    """
    chunks = retrieve(query, k=k)
    if not is_in_scope(chunks):
        return GroundedAnswer(answer=OUT_OF_SCOPE_MESSAGE, citations=[], refused=True)

    answer_text = " ".join(chunk.text for chunk in chunks)
    citations = [{"title": c.source_title, "url": c.source_url} for c in chunks]
    return GroundedAnswer(answer=answer_text, citations=citations, refused=False)


if __name__ == "__main__":
    for query in [
        "What does the ABCDE rule mean for checking moles?",
        "What's the weather like in Sydney today?",
    ]:
        result = answer_query(query, k=3)
        print(f"\nQuery: {query}")
        print(f"Refused: {result.refused}")
        print(f"Answer: {result.answer[:150]}...")
        print(f"Citations: {[c['title'] for c in result.citations]}")
