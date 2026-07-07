"""
LangGraph wiring for the SkinScope AU agent.

Linear pipeline per the build plan: check_scope -> segment -> classify ->
retrieve -> classify -> compose -> safety_gate. Every path ends at
safety_gate, which is the only node allowed to construct the final
AgentResponse — see agent/nodes.py for why that makes the disclaimer
non-bypassable.

Segment/classify run unconditionally in sequence (not gated on in_scope):
an out-of-scope question shouldn't block a legitimate image upload from
getting scored — only the text answer gets refused, inside safety_gate.
"""

from langgraph.graph import END, StateGraph

from agent.nodes import (
    check_scope_node,
    classify_node,
    compose_node,
    retrieve_node,
    safety_gate_node,
    segment_node,
)
from agent.schemas import AgentResponse, AgentState
from agent.tracing import trace_run


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("check_scope", check_scope_node)
    graph.add_node("segment", segment_node)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("compose", compose_node)
    graph.add_node("safety_gate", safety_gate_node)

    graph.set_entry_point("check_scope")
    graph.add_edge("check_scope", "segment")
    graph.add_edge("segment", "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "compose")
    graph.add_edge("compose", "safety_gate")
    graph.add_edge("safety_gate", END)

    return graph.compile()


_compiled_graph = None


def get_agent():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(state_input: dict) -> AgentResponse:
    """
    The single entry point external callers (FastAPI, tests) should use.
    Wraps the compiled graph with tracing so every real invocation is logged,
    without needing tracing logic inside the nodes themselves.
    """
    app = get_agent()
    result = app.invoke(state_input)
    response = result["response"]
    trace_run(
        question=state_input.get("question"),
        image_present=bool(state_input.get("image_bytes")),
        response=response,
    )
    return response


if __name__ == "__main__":
    app = build_graph()

    print("--- Text-only, in-scope question ---")
    result = app.invoke({"question": "What is the ABCDE rule for checking moles?"})
    response = result["response"]
    print("Refused:", response.refused)
    print("Answer:", response.answer[:150], "...")
    print("Citations:", [c.title for c in response.citations])
    print("Disclaimer present:", bool(response.disclaimer))

    print("\n--- Text-only, out-of-scope question ---")
    result = app.invoke({"question": "What's the weather like in Sydney today?"})
    response = result["response"]
    print("Refused:", response.refused)
    print("Answer:", response.answer)
    print("Disclaimer present:", bool(response.disclaimer))