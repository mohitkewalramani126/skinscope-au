"""
Retriever for SkinScope AU's RAG corpus.

Retrieves the top-k most relevant chunks from the Chroma vector store built by
build_index.py, using the same sentence-transformers embedding model so query
and corpus embeddings live in the same space.

Also enforces an out-of-scope gate: if the best match isn't similar enough to
the query, the corpus has nothing relevant to say, and the caller should refuse
rather than force an answer out of a weak match.

Threshold tuning (evaluation/rag_eval.py, swept against evaluation/golden_qa.json,
22 questions / 17 in-scope, 5 out-of-scope):

  threshold  scope_accuracy
  0.50-0.65  100.00%
  0.70-0.75   90.91%  (a "what's the weather in Sydney" query leaks through —
                       UV Index content sits closer to generic weather queries
                       in embedding space than expected)

0.65 is used: the widest margin that still holds 100% scope accuracy on the
golden set.
"""

import os
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

RAG_DIR = os.path.dirname(__file__)
CHROMA_DB_PATH = os.path.join(RAG_DIR, "chroma_db")
COLLECTION_NAME = "skinscope_corpus"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chroma's cosine "distance" is (1 - cosine_similarity), so smaller = more similar.
# Value chosen empirically — see module docstring for the threshold sweep.
OUT_OF_SCOPE_DISTANCE_THRESHOLD = 0.65

_model: SentenceTransformer | None = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source_title: str
    source_url: str
    distance: float


def retrieve(query: str, k: int = 3) -> list[RetrievedChunk]:
    """Return the top-k chunks for `query`, ranked by cosine distance (ascending)."""
    embedding = _get_model().encode([query]).tolist()
    results = _get_collection().query(query_embeddings=embedding, n_results=k)

    chunks = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        chunks.append(RetrievedChunk(
            id=results["ids"][0][i],
            text=results["documents"][0][i],
            source_title=metadata["source_title"],
            source_url=metadata["source_url"],
            distance=results["distances"][0][i],
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
    "relevant information to answer that — please ask something related to skin "
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
    verbatim) — there is no LLM here yet. Day 11 will wrap this exact contract
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