"""
Tracing for the SkinScope AU agent, via Langfuse.

One function, called once per agent run in agent/graph.py's run_agent(), given
the inputs and the final response. Tracing failures never break the agent
response itself — a bad/missing Langfuse key should degrade to "no trace
recorded", not a 500 to the end user.
"""

import os

from agent.schemas import AgentResponse

_client = None


def _get_langfuse_client():
    """Lazily construct the Langfuse client, only if credentials are present."""
    global _client
    if _client is not None:
        return _client
    if not os.environ.get("LANGFUSE_SECRET_KEY") or not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    try:
        from langfuse import get_client
    except ImportError:
        # Day 14: langfuse is no longer in requirements.txt (its opentelemetry
        # dependency chain was ~13MB, part of trimming the Vercel bundle under
        # the 500MB function limit). If someone sets the Langfuse env vars
        # without the package installed, degrade to "no trace recorded" rather
        # than crashing every request -- same never-break-the-response
        # philosophy as trace_run()'s own except clause below.
        return None

    _client = get_client()
    return _client


def trace_run(question: str | None, image_present: bool, response: AgentResponse) -> None:
    """Log one agent run as a Langfuse span. No-ops if Langfuse isn't configured."""
    client = _get_langfuse_client()
    if client is None:
        return

    try:
        with client.start_as_current_observation(
            as_type="span", name="skinscope-agent-run"
        ) as span:
            span.update(
                input={"question": question, "image_present": image_present},
                output={
                    "refused": response.refused,
                    "risk_band": response.risk_band,
                    "risk_score": response.risk_score,
                    "escalate": response.escalate,
                    "has_answer": response.answer is not None,
                    "n_citations": len(response.citations),
                },
            )
        client.flush()
    except Exception:
        # tracing must never take down a real request
        pass
