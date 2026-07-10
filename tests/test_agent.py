
"""
Tests for the SkinScope AU agent — focused on the safety-critical, deterministic
parts: the safety gate (disclaimer/refusal/escalation logic) and the scope gate.

Deliberately NOT covered here: compose_node's real Groq call. Hitting a live
LLM API on every `pytest -q` run would make the suite slow, cost money, and be
flaky (network issues, rate limits) — a real tradeoff, not an oversight. The
no-key fallback path (extractive answer) is exercised indirectly since it's
plain string logic with no I/O. Groq's actual output quality is checked
separately in evaluation/agent_eval.py (LLM-as-judge faithfulness), which is
meant to be run deliberately, not on every test invocation.
"""

from agent.nodes import safety_gate_node, check_scope_node
from agent.schemas import AgentState


def test_safety_gate_always_includes_disclaimer_no_input():
    state: AgentState = {}
    result = safety_gate_node(state)
    response = result["response"]
    assert response.disclaimer  # non-empty, present unconditionally


def test_safety_gate_always_includes_disclaimer_in_scope_answer():
    state: AgentState = {
        "question": "What is the ABCDE rule?",
        "in_scope": True,
        "composed_answer": "Some grounded answer.",
        "citations": [{"title": "Source", "url": "https://example.com"}],
    }
    result = safety_gate_node(state)
    response = result["response"]
    assert response.disclaimer
    assert response.refused is False
    assert response.answer == "Some grounded answer."
    assert len(response.citations) == 1


def test_safety_gate_refuses_out_of_scope_question():
    state: AgentState = {
        "question": "What's the weather like today?",
        "in_scope": False,
    }
    result = safety_gate_node(state)
    response = result["response"]
    assert response.refused is True
    assert response.citations == []
    assert response.disclaimer  # refusal still carries the disclaimer


def test_safety_gate_vision_results_survive_an_out_of_scope_question():
    # the core Day-11 design decision: an off-topic question must not suppress
    # a legitimate image analysis riding alongside it
    state: AgentState = {
        "question": "What's the weather like today?",
        "in_scope": False,
        "mask_png_base64": "fake_base64_mask",
        "risk_score": 0.2,
        "risk_band": "low",
    }
    result = safety_gate_node(state)
    response = result["response"]
    assert response.refused is True
    assert response.risk_band == "low"
    assert response.risk_score == 0.2
    assert response.mask_png_base64 == "fake_base64_mask"


def test_safety_gate_escalates_on_high_risk_band():
    state: AgentState = {"risk_band": "high"}
    result = safety_gate_node(state)
    response = result["response"]
    assert response.escalate is True
    assert response.escalation_reason is not None


def test_safety_gate_no_escalation_on_low_risk_band():
    state: AgentState = {"risk_band": "low"}
    result = safety_gate_node(state)
    response = result["response"]
    assert response.escalate is False
    assert response.escalation_reason is None


def test_check_scope_node_in_scope_question():
    state: AgentState = {"question": "What is the ABCDE rule for spotting melanoma?"}
    result = check_scope_node(state)
    assert result["in_scope"] is True
    assert len(result["retrieved_chunks"]) > 0


def test_check_scope_node_out_of_scope_question():
    state: AgentState = {"question": "What's the weather like in Sydney today?"}
    result = check_scope_node(state)
    assert result["in_scope"] is False
    assert result["retrieved_chunks"] == []


def test_check_scope_node_no_question_defaults_in_scope():
    # no question asked -> nothing to refuse; image-only requests must proceed
    state: AgentState = {}
    result = check_scope_node(state)
    assert result["in_scope"] is True


# ---------- sensitivity_mode (Day 12 feature): standard vs. high ----------
# standard escalates only on "high" (50.7% sensitivity / 95.1% specificity);
# "high" also escalates on "moderate" (76.7% sensitivity / 86.1% specificity).
# Both operating points are measured, not invented -- see docs/model_card.md.

def test_standard_mode_does_not_escalate_on_moderate_band():
    state: AgentState = {"risk_band": "moderate", "sensitivity_mode": "standard"}
    response = safety_gate_node(state)["response"]
    assert response.escalate is False
    assert response.escalation_reason is None
    assert response.sensitivity_mode == "standard"


def test_high_sensitivity_mode_escalates_on_moderate_band():
    state: AgentState = {"risk_band": "moderate", "sensitivity_mode": "high"}
    response = safety_gate_node(state)["response"]
    assert response.escalate is True
    assert response.escalation_reason is not None
    assert response.sensitivity_mode == "high"


def test_high_sensitivity_mode_still_escalates_on_high_band():
    state: AgentState = {"risk_band": "high", "sensitivity_mode": "high"}
    response = safety_gate_node(state)["response"]
    assert response.escalate is True


def test_high_sensitivity_mode_does_not_escalate_on_low_band():
    state: AgentState = {"risk_band": "low", "sensitivity_mode": "high"}
    response = safety_gate_node(state)["response"]
    assert response.escalate is False


def test_sensitivity_mode_defaults_to_standard_when_absent():
    # no sensitivity_mode key at all -> must default safely to "standard",
    # not silently escalate more than a client explicitly asked for
    state: AgentState = {"risk_band": "moderate"}
    response = safety_gate_node(state)["response"]
    assert response.escalate is False
    assert response.sensitivity_mode == "standard"