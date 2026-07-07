"""
Evaluates the RAG retriever against a hand-authored golden Q&A set.

Three things measured, each tied to a real production risk for a cited
awareness tool:
  - precision@k:      of the chunks we retrieve, how many are actually relevant?
                       (low precision -> irrelevant citations shown to users)
  - citation coverage: for in-scope questions, did the right source show up
                       anywhere in the top-k at all? (misses -> silently wrong answers)
  - scope accuracy:    does the distance-based scope gate correctly refuse
                       genuinely unrelated questions without also refusing
                       legitimate ones? Threshold is swept here, not guessed
                       (see the weather-query miss with threshold=0.75).

Faithfulness (Day 10 scope): answers at this stage are extractive — the
"answer" is just retrieved chunk text plus its citation, nothing is generated.
So faithfulness here checks a narrower, still-real thing: every sentence in
the composed answer is verbatim from a retrieved, cited chunk (i.e. the
compose step introduced nothing unsupported). A real generative faithfulness
check (LLM-as-judge, hallucination detection) is Day 11 territory, once Groq
phrasing is wired on top of this retriever.
"""

import json
import os

from rag.retriever import RetrievedChunk, retrieve

EVAL_DIR = os.path.dirname(__file__)
GOLDEN_QA_PATH = os.path.join(EVAL_DIR, "golden_qa.json")
K = 3


def compose_answer(chunks: list[RetrievedChunk]) -> str:
    """Extractive answer: concatenate retrieved chunk text verbatim, cited."""
    return " ".join(chunk.text for chunk in chunks)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(retrieved_ids)


def citation_covered(retrieved_ids: list[str], relevant_ids: set[str]) -> bool:
    return any(rid in relevant_ids for rid in retrieved_ids)


def faithfulness(answer: str, chunks: list[RetrievedChunk]) -> float:
    """Fraction of the answer's sentences that appear verbatim in a cited chunk."""
    source_text = " ".join(chunk.text for chunk in chunks)
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if not sentences:
        return 1.0
    supported = sum(1 for s in sentences if s in source_text)
    return supported / len(sentences)


def evaluate(distance_threshold: float):
    with open(GOLDEN_QA_PATH) as f:
        golden = json.load(f)

    precisions, coverages, faithfulnesses = [], [], []
    scope_correct = 0

    for item in golden:
        chunks = retrieve(item["question"], k=K)
        retrieved_ids = [c.id for c in chunks]
        predicted_in_scope = bool(chunks) and chunks[0].distance <= distance_threshold

        if predicted_in_scope == item["in_scope"]:
            scope_correct += 1

        if item["in_scope"]:
            relevant = set(item["relevant_chunk_ids"])
            precisions.append(precision_at_k(retrieved_ids, relevant))
            coverages.append(citation_covered(retrieved_ids, relevant))
            answer = compose_answer(chunks)
            faithfulnesses.append(faithfulness(answer, chunks))

    return {
        "threshold": distance_threshold,
        "scope_accuracy": scope_correct / len(golden),
        "mean_precision_at_k": sum(precisions) / len(precisions),
        "citation_coverage": sum(coverages) / len(coverages),
        "mean_faithfulness": sum(faithfulnesses) / len(faithfulnesses),
    }


if __name__ == "__main__":
    print(f"{'threshold':>10} {'scope_acc':>10} {'precision@k':>12} {'coverage':>9} {'faithfulness':>13}")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        r = evaluate(t)
        print(
            f"{r['threshold']:>10.2f} {r['scope_accuracy']:>10.2%} "
            f"{r['mean_precision_at_k']:>12.2%} {r['citation_coverage']:>9.2%} "
            f"{r['mean_faithfulness']:>13.2%}"
        )