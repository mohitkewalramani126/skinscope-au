"""
LLM-as-judge faithfulness evaluation for the Day 11 agent's Groq-composed answers.

Day 10's rag_eval.py checked faithfulness on extractive answers (trivially 100%,
since verbatim retrieved text can't contain unsupported claims by construction).
This checks the real thing: does the LLM-generated answer only state what the
cited sources actually say? A second, independent Groq call acts as judge,
reading the same sources and the generated answer, and flags any claim not
directly supported.

Includes a small human-checked set: every result is printed with the full
answer, sources, and the judge's reasoning, so a person can read a sample and
confirm or challenge the judge's calls — LLM judges have their own failure
modes (they can be fooled by confident-sounding unsupported claims, or be
overly strict on reasonable paraphrasing) and shouldn't be trusted blind.
"""

import json
import os

from groq import Groq
from pydantic import BaseModel

from agent.graph import build_graph
from rag.retriever import retrieve

GROQ_MODEL = "llama-3.1-8b-instant"

JUDGE_PROMPT_TEMPLATE = """You are a strict fact-checker. You will be given SOURCES and an ANSWER that claims to be based only on those sources.

Your job: identify any claim in the ANSWER that is NOT directly supported by the SOURCES. Minor rephrasing or summarizing is fine and does not count as unsupported. Only flag claims that introduce information, numbers, or advice not present in the sources.

SOURCES:
{sources}

ANSWER:
{answer}

Respond with ONLY a JSON object in this exact shape, no other text:
{{"faithful": true or false, "unsupported_claims": ["...", ...], "reasoning": "one sentence"}}
"""


class JudgeVerdict(BaseModel):
    faithful: bool
    unsupported_claims: list[str]
    reasoning: str


def judge_faithfulness(answer: str, sources_text: str) -> JudgeVerdict:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    prompt = JUDGE_PROMPT_TEMPLATE.format(sources=sources_text, answer=answer)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    data = json.loads(completion.choices[0].message.content)
    return JudgeVerdict(**data)


def run_eval():
    with open("evaluation/golden_qa.json") as f:
        golden = json.load(f)
    in_scope_questions = [item["question"] for item in golden if item["in_scope"]]

    app = build_graph()
    results = []
    for question in in_scope_questions:
        state = app.invoke({"question": question})
        response = state["response"]
        if response.refused or not response.answer:
            print(f"SKIPPED (refused unexpectedly): {question}")
            continue

        chunks = retrieve(question, k=3)
        sources_text = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
        verdict = judge_faithfulness(response.answer, sources_text)
        results.append({"question": question, "answer": response.answer, "verdict": verdict})

    return results


if __name__ == "__main__":
    results = run_eval()

    print("=" * 80)
    print("FULL RESULTS (human-checkable set)")
    print("=" * 80)
    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer']}")
        print(f"Faithful: {r['verdict'].faithful}")
        if r["verdict"].unsupported_claims:
            print(f"Unsupported claims: {r['verdict'].unsupported_claims}")
        print(f"Judge reasoning: {r['verdict'].reasoning}")

    faithful_count = sum(1 for r in results if r["verdict"].faithful)
    print("\n" + "=" * 80)
    print(f"SUMMARY: {faithful_count}/{len(results)} answers judged faithful "
          f"({faithful_count / len(results):.1%})")