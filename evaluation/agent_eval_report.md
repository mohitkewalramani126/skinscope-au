# Agent Faithfulness Evaluation (Day 11)

Evaluates the actual Groq-generated answers from the full LangGraph agent — not
the extractive Day 10 answers, which are trivially faithful by construction
(verbatim retrieved text can't contain unsupported claims). This is the real
test: does an LLM, prompted to answer only from given sources, actually stay
grounded?

## Method

All 22 in-scope questions from `evaluation/golden_qa.json` were run through the
full agent (`agent/graph.py`). For each answer produced, a second, independent
Groq call acted as judge: given the same 3 retrieved source chunks and the
generated answer, it flagged any claim not directly supported by those sources.
Full transcripts (question, answer, verdict, reasoning) were printed for every
result — a "human-checked set" in the fullest sense, not a sample — because
LLM judges have their own failure modes and shouldn't be trusted blind.

## Results

- 2 of 22 questions were refused by the scope gate before reaching the agent
  ("What does ABCDE mean?", "Am I at risk?") — this is the exact same blind
  spot documented in `evaluation/rag_eval_report.md` (extremely short,
  context-free questions), not a new failure. 20 questions were actually
  answered and judged.
- **LLM-judge faithfulness: 16/20 (80.0%)**
- **Human-corrected faithfulness: 18/20 (90.0%)** — see below for why 2 of the
  judge's 4 flags were overruled on manual review.

## Manual review of all 4 flagged answers

| # | Question | Judge's flag | My read | Verdict |
|---|---|---|---|---|
| 1 | "When should I get a new mole checked by a doctor?" | Added "new moles are less common after 25" as a rationale not stated in the source | The source says "see your doctor if a new mole appears after 25" but never explains *why* — the model invented a plausible-sounding reason. This is a real hallucination, small but real. | **Confirmed unfaithful** |
| 2 | "What do the different skin types mean for sun sensitivity?" | Used "moderately sensitive" / "somewhat resistant to burning" — not verbatim source terms | The source describes Type II as "usually burns and sometimes tans" and Type III as "sometimes burns and usually tans." The model's wording is a reasonable paraphrase of the same underlying gradient, not new information. | **Judge too strict — overruled** |
| 3 | "Is it safe to get a base tan before summer?" | Flagged "not safe... leads to UV damage and skin cancer" as unsupported | The source explicitly says "there is no safe way to tan" and directly connects tanning to sun-damaged cells and melanoma risk. The model's claim is a correct, direct summary of that same content. | **Judge too strict — overruled** |
| 4 | "What increases a person's risk of getting skin cancer?" | Answer included Australia-wide statistics (two-in-three lifetime risk, 95-99% sun-caused, BCC/SCC 99% split) not present in the 3 chunks retrieved for this question | Confirmed: those facts live in a *different* corpus chunk (`non-melanoma-stats`) that wasn't retrieved for this query. The model pulled them from its own training data, not from what it was actually shown — true facts, wrong grounding. This is the real risk faithfulness-checking exists to catch. | **Confirmed unfaithful — most serious finding** |

## Why this matters

Case 4 is the important one: the model produced **correct, true statements**
that happen to also exist in the corpus — but not in the specific context it
was given for that question. A naive "is this true?" check would have passed
it. Faithfulness has to mean "grounded in what was actually retrieved," not
"factually accurate in general," or exactly this kind of confident,
plausible-looking, ungrounded answer slips through.

## Known limitations

- Sample size is small (20 judged answers) — a real production system would
  want a larger, more diverse judged set before trusting an 80-90% faithfulness
  number as stable.
- The judge model (llama-3.1-8b-instant) is the same model family used for
  answer composition. A stronger or differently-trained judge model might
  disagree with some verdicts — using the same model to grade its own family's
  output is a known limitation of LLM-as-judge setups.
- Manual review (this document) is itself a single reviewer's read: an
  independent second human reviewer might score cases 2 and 3 differently.
- The same 2-question scope-gate blind spot from Day 10/11's retriever
  evaluation reappears here — see `evaluation/rag_eval_report.md`.
