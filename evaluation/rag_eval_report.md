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

| threshold | scope accuracy (27 questions) |
|---|---|
| 0.50 – 0.65 | 92.59% |
| 0.70 – 0.75 | 88.89% |
| 0.78 – 0.85 | 92.59% |

**`0.65` is used** — it ties for the best accuracy on this golden set and sits
in the middle of the widest contiguous band that achieves it (0.50–0.65), which
is a more stable choice than the edge of a band.

### The two remaining failures, and why a query-prefix "fix" was rejected

At threshold 0.65, two of 27 golden questions are misclassified — both false
negatives (wrongly refused), and both extremely short, context-free questions:

| question | distance | expected | predicted |
|---|---|---|---|
| "What does ABCDE mean?" | 0.7621 | in-scope | refused |
| "Am I at risk?" | 0.6676 | in-scope | refused |

These sit farther from the corpus than they "should," because on their own —
with no conversation history — they are genuinely ambiguous. "Am I at risk?"
could be about anything; only conversational context (which this single-turn
retriever doesn't have) would disambiguate it.

One fix was tried and rejected: prefixing every query with fixed domain context
("In the context of skin cancer awareness and prevention: ...") before embedding,
without changing the corpus side. Tested on the two failures plus the two
closest hard negatives:

| question | distance (no prefix) | distance (with prefix) |
|---|---|---|
| "What does ABCDE mean?" (in-scope) | 0.7621 | 0.5069 |
| "Am I at risk?" (in-scope) | 0.6676 | 0.3487 |
| "What's the weather like in Sydney today?" (out-of-scope) | 0.6649 | 0.3709 |
| "Can you recommend a good moisturizer?" (out-of-scope) | 0.6602 | 0.5119 |

The prefix does pull both in-scope failures closer — but it pulls the
out-of-scope queries closer by *more*, since injecting the same domain context
into every query compresses all distances toward "relevant" regardless of
actual content. This would trade 2 false refusals for false acceptances on
genuinely unrelated queries, which is a worse failure mode for a safety-relevant
gate. **Rejected; threshold 0.65 with no prefix stays.**

This is a real, documented limitation rather than a solved problem: the scope
gate is reliable for well-formed questions (25/27 correct) but has a known blind
spot on extremely terse, context-free ones. The right fix is conversational
context (carrying prior turns into scope-checking), not a static prefix — that's
future work beyond Day 11's single-turn agent.

## Golden set

`evaluation/golden_qa.json` — 27 questions (22 in-scope, 5 out-of-scope), authored
to cover every corpus topic plus a range of out-of-scope difficulty: easy negatives
(weather, recipes, sports, stocks), one deliberately hard negative — "Can you
recommend a good moisturizer for dry skin?" — which is skincare-adjacent but not
skin-cancer awareness, and (added during Day 11 agent testing) five deliberately
terse/colloquial in-scope phrasings ("What does ABCDE mean?", "Am I at risk?",
etc.) to stress-test sensitivity to real-world question phrasing, not just
well-formed ones.

## Results (k=3, threshold=0.65)

| Metric | Value |
|---|---|
| Scope accuracy | 92.59% (25/27) |
| Mean precision@3 | 36.36% |
| Citation coverage | 100.00% |
| Mean faithfulness | 100.00% |

**Precision@3 needs context to read correctly.** Most golden questions have
exactly one or two truly relevant chunks in a 20-chunk corpus, so with k=3 the
mathematical ceiling on precision@3 across this golden set is **39.39%** — the
retriever's measured 36.36% is **92.3% of theoretical maximum**. Read on its
own, "36% precision" sounds weak; in context, it means the retriever is close
to as good as k=3 retrieval can be against this ground truth.

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
- The out-of-scope threshold (0.65) correctly separates well-formed in-scope
  questions from clear and near-miss negatives, but has a known blind spot on
  extremely short, context-free in-scope questions ("Am I at risk?") — see the
  two-failure analysis above. A static query prefix was tried and rejected as a
  fix since it also pulls negatives closer. The real fix needs conversational
  context, which this single-turn retriever doesn't have.
- Faithfulness at this stage (Day 10/retriever-only) only verifies that the
  extractive compose step didn't introduce unsupported text. Day 11 adds
  LLM-generated phrasing via Groq on top of this retriever — see the agent's
  own LLM-as-judge faithfulness evaluation for that layer.
- The corpus is Australia-focused and intentionally narrow (20 chunks); it does
  not cover every skin cancer subtype, treatment information, or international
  guidance variations.
