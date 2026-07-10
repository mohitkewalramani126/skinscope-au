"""
Node functions for the SkinScope AU LangGraph agent.

Each node takes the current AgentState and returns a dict of state updates
(LangGraph merges these in). Nodes wrap existing, already-tested code from
Day 9 (vision/inference.py) and Day 10 (rag/retriever.py) rather than
reimplementing anything — this graph is an orchestration layer, not a new
inference layer.

Design note on `retrieve_node`: determining in-scope status *requires* the
retrieval distance in the first place (is_in_scope reads chunks[0].distance),
so the actual retrieval call happens inside check_scope_node. retrieve_node
is a deliberate no-op/pass-through here — it exists as a named seam matching
the plan's graph shape, and as the place future changes (re-ranking, MMR
diversity, fetching more chunks) would go, without duplicating the Chroma
query that already ran.

Design note on scope + vision together: if a user uploads an image AND asks
an off-topic question, the risk score/mask should still come through — only
the text answer gets refused. So `in_scope` only gates the answer/citations,
never the vision fields.
"""

import os

import cv2
import numpy as np
from groq import Groq

from agent.schemas import AgentResponse, AgentState, Citation
from rag.retriever import OUT_OF_SCOPE_MESSAGE, is_in_scope, retrieve
from vision.inference import (
    CLASSIFIER_TEMPERATURE,
    _mask_to_png_base64,
    _risk_band,
    _sigmoid,
    get_classifier_session,
    get_segmentation_session,
    load_image_rgb,
    preprocess_for_classifier,
    preprocess_for_segmentation,
)

GROQ_MODEL = "llama-3.1-8b-instant"  # confirmed active production model, console.groq.com/docs/models

DISCLAIMER = (
    "SkinScope AU is an awareness tool, not a medical device. It does not diagnose "
    "skin cancer. Always have any new, changing, or concerning spot checked by a "
    "doctor or dermatologist."
)
ESCALATION_MESSAGE = (
    "This result suggests a higher-risk pattern. Please see a doctor or dermatologist "
    "for a proper assessment as soon as you can. Do not rely on this tool's output alone."
)
HIGH_SENSITIVITY_MODERATE_MESSAGE = (
    "You've enabled high-sensitivity mode, which also flags moderate-signal results for "
    "an in-person check. At the standard setting, this tool misses about half of real "
    "cases; this mode catches more of them at the cost of more false alarms. Please see "
    "a doctor or dermatologist for a proper assessment. Do not rely on this tool's "
    "output alone."
)


def check_scope_node(state: AgentState) -> dict:
    question = state.get("question")
    if not question:
        # nothing was asked, so there's nothing to be out of scope about
        return {"in_scope": True, "retrieved_chunks": []}
    chunks = retrieve(question, k=3)
    in_scope = is_in_scope(chunks)
    return {"in_scope": in_scope, "retrieved_chunks": chunks if in_scope else []}


def segment_node(state: AgentState) -> dict:
    image_bytes = state.get("image_bytes")
    if not image_bytes:
        return {}
    image_rgb = load_image_rgb(image_bytes)
    original_h, original_w = image_rgb.shape[:2]
    seg_input = preprocess_for_segmentation(image_rgb)
    seg_logits = get_segmentation_session().run(None, {"input": seg_input})[0]
    seg_probs = _sigmoid(seg_logits[0, 0])
    mask_small = (seg_probs > 0.5).astype(np.uint8)
    mask_full_size = cv2.resize(
        mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST
    )
    return {"mask_png_base64": _mask_to_png_base64(mask_full_size)}


def classify_node(state: AgentState) -> dict:
    image_bytes = state.get("image_bytes")
    if not image_bytes:
        return {}
    image_rgb = load_image_rgb(image_bytes)
    cls_input = preprocess_for_classifier(image_rgb)
    cls_logit = float(get_classifier_session().run(None, {"input": cls_input})[0][0][0])
    calibrated_score = float(_sigmoid(np.array(cls_logit / CLASSIFIER_TEMPERATURE)))
    return {
        "risk_score": calibrated_score,
        "risk_band": _risk_band(calibrated_score),
        "raw_logit": cls_logit,
    }


def retrieve_node(state: AgentState) -> dict:
    """No-op by design — see module docstring."""
    return {}


def compose_node(state: AgentState) -> dict:
    question = state.get("question")
    if not question or not state.get("in_scope"):
        return {"composed_answer": None, "citations": []}

    chunks = state.get("retrieved_chunks", [])
    citations = [{"title": c.source_title, "url": c.source_url} for c in chunks]
    source_text = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        # no-key fallback: same extractive behaviour as Day 10
        return {"composed_answer": " ".join(c.text for c in chunks), "citations": citations}

    prompt = (
        "Answer the user's question using ONLY the information in the sources below. "
        "Do not add any fact that isn't in the sources. If the sources don't fully "
        "answer the question, say so rather than guessing.\n\n"
        f"Sources:\n{source_text}\n\n"
        f"Question: {question}\n\n"
        "Answer in 2-4 plain-language sentences."
    )
    try:
        client = Groq(api_key=groq_api_key)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception:
        # Groq call failed (network, rate limit, bad key) -> fall back rather than crash
        answer = " ".join(c.text for c in chunks)

    return {"composed_answer": answer, "citations": citations}


def safety_gate_node(state: AgentState) -> dict:
    """
    The non-bypassable node. Every path through the graph ends here, and this
    is the only place AgentResponse gets constructed — so no other node can
    produce a final response without a disclaimer attached.
    """
    question = state.get("question")
    in_scope = state.get("in_scope", True)
    risk_band = state.get("risk_band")
    sensitivity_mode = state.get("sensitivity_mode", "standard")

    if question and not in_scope:
        answer = OUT_OF_SCOPE_MESSAGE
        citations = []
        refused = True
    else:
        answer = state.get("composed_answer")
        citations = [Citation(**c) for c in state.get("citations", [])]
        refused = False

    if sensitivity_mode == "high":
        # escalate on "moderate" or "high" -- 76.7% sensitivity / 86.1%
        # specificity (measured Day 7, threshold 0.5). More false alarms,
        # catches more real cases.
        escalate = risk_band in ("moderate", "high")
        escalation_reason = HIGH_SENSITIVITY_MODERATE_MESSAGE if risk_band == "moderate" else ESCALATION_MESSAGE
    else:
        # escalate only on "high" -- 50.7% sensitivity / 95.1% specificity
        # (measured Day 7, threshold 0.7918). Fewer false alarms, misses more.
        escalate = risk_band == "high"
        escalation_reason = ESCALATION_MESSAGE

    response = AgentResponse(
        mask_png_base64=state.get("mask_png_base64"),
        risk_score=state.get("risk_score"),
        risk_band=risk_band,
        answer=answer,
        citations=citations,
        disclaimer=DISCLAIMER,
        refused=refused,
        escalate=escalate,
        escalation_reason=escalation_reason if escalate else None,
        sensitivity_mode=sensitivity_mode,
    )
    return {"response": response}