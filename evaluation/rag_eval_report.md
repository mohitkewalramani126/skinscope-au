# RAG Retrieval Evaluation

Covers the retrieval and grounding layer for SkinScope AU's awareness Q&A: corpus
construction, embedding/index setup, the out-of-scope refusal gate, and evaluation
against a hand-authored golden set. All numbers below are measured, not estimated.

## Corpus

`rag/corpus.json` — 20 chunks, each a single topic-scoped paragraph (one idea per
chunk: the ABCDE rule, one skin cancer type, one prevention step, etc.) rather than
arbitrary fixed-length windows over a larger document. Every chunk was manually
verified against its live source page (fetched directly, not from a search summary)
and carries the exact source title and URL:

- Cancer Council Australia — "Check for signs of skin cancer", "UV Index",
  "Basal and Squamous Cell Carcinoma | Non-melanoma Skin Cancer"
- healthdirect.gov.au — "Should I be checked for skin cancer?"
- Melanoma Institute Australia — "Checking Your Skin", "How to Prevent Melanoma"

One candidate statistic (a Cancer Australia incidence figure) was deliberately
excluded: the source page returned no fetchable content (JavaScript-rendered), so
the number could only be traced to an unverified web-search summary. It was left
out rather than risk an uncited or misattributed statistic. All 20 chunks in the
corpus are backed by content actually fetched and read in full.

## Index

`rag/build_index.py` embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2`
and stores them in a local persistent Chroma collection (cosine distance space).
Re-running this script rebuilds the index from scratch, so it stays reproducible
as the corpus changes.

**Dependency note:** the environment's torch (2.2.2) can't be upgraded (no newer
wheel available), and the latest `sentence-transformers`/`transformers` releases
now hard-require torch>=2.4. Pinned `sentence-transformers==2.7.0` (contemporaneous
with torch 2.2) resolves this. Also pinned `opencv-python-headless==4.10.0.84` and
`numpy<2`, since newer opencv wheels require numpy>=2 while this torch version
requires numpy<2 — both constraints are recorded in `requirements.txt` so a fresh
install doesn't silently drift into the same conflict.

## Retriever and grounding

`rag/retriever.py` exposes:

- `retrieve(query, k)` — top-k chunks by cosine distance.
- `is_in_scope(chunks)` — scope gate: refuses if the best match is farther than
  `OUT_OF_SCOPE_DISTANCE_THRESHOLD`.
- `answer_query(query, k)` — the actual grounding + refusal contract: returns a
  cited, extractive answer if in scope, or a fixed refusal message with no
  citations if not.

Day 10's answer is extractive (retrieved chunk text, concatenated verbatim) — no
LLM is involved yet. Day 11 wraps this same contract (same citations list, same
refusal gate) with Groq to phrase answers naturally, without changing what counts
as grounded or in-scope.

## Out-of-scope threshold: swept, not guessed

An early manual check at threshold 0.75 let a "What's the weather like in Sydney
today?" query through as in-scope (distance 0.665) — UV Index content sits closer
to generic weather queries in embedding space than expected. Rather than adjust
by eye, `evaluation/rag_eval.py` sweeps the threshold against the full golden set:

| threshold | scope accuracy |
|---|---|
| 0.50 – 0.65 | 100.00% |
| 0.70 – 0.75 | 90.91% (the weather query leaks through) |

**`0.65` is used** — the widest margin that still holds 100% scope accuracy on
this golden set.

## Golden set

`evaluation/golden_qa.json` — 22 questions (17 in-scope, 5 out-of-scope), authored
to cover every corpus topic plus a range of out-of-scope difficulty: easy negatives
(weather, recipes, sports, stocks) and one deliberately hard negative — "Can you
recommend a good moisturizer for dry skin?" — which is skincare-adjacent but not
skin-cancer awareness, to stress-test the scope gate against topically-near
distractors, not just obviously unrelated ones.

## Results (k=3, threshold=0.65)

| Metric | Value |
|---|---|
| Scope accuracy | 100.00% |
| Mean precision@3 | 35.29% |
| Citation coverage | 100.00% |
| Mean faithfulness | 100.00% |

**Precision@3 needs context to read correctly.** Most golden questions (15 of 17
in-scope) have exactly one truly relevant chunk in the corpus, so with k=3 the
mathematical ceiling on precision@3 is 33.3% for those questions (66.7% for the
2 questions with 2 relevant chunks). Computed across the golden set, the ceiling
is **37.25%** — the retriever's measured 35.29% is **94.7% of theoretical maximum**.
Read on its own, "35% precision" sounds weak; in context, it means the retriever
is close to as good as k=3 retrieval can be against this ground truth.

**Citation coverage of 100%** is the more informative number here: for every
in-scope golden question, the correct source appeared somewhere in the top-3
results — the retriever never missed the right document, it just also surfaced
adjacent (imperfectly relevant) chunks alongside it, which the precision ceiling
above already explains.

**Faithfulness of 100%** is expected, not a strong claim about generation quality:
the Day 10 answer is extractive (verbatim retrieved text), so by construction it
cannot contain unsupported sentences. This metric becomes meaningful once Day 11
adds LLM-generated phrasing on top — that will need its own faithfulness check
against actually-generated (not merely concatenated) text.

## Known limitations

- Precision@3 is capped low by golden-set structure (mostly single-relevant-chunk
  ground truth), not purely retriever weakness — see ceiling calculation above.
- The out-of-scope threshold (0.65) is tuned against a 22-question golden set;
  it has not been stress-tested against a larger or more adversarial set of
  near-miss queries (e.g., cosmetic dermatology, general "is this normal" health
  questions unrelated to skin cancer).
- Faithfulness at this stage only verifies that the extractive compose step
  didn't introduce unsupported text — it does not yet test hallucination
  resistance under LLM-generated phrasing, which is Day 11 scope.
- The corpus is Australia-focused and intentionally narrow (20 chunks); it does
  not cover every skin cancer subtype, treatment information, or international
  guidance variations.
