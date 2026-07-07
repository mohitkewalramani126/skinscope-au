"""
Structured schemas for the SkinScope AU agent.

The agent takes an optional image and/or question, runs it through a
LangGraph pipeline (check_scope -> segment -> classify -> retrieve -> compose
-> safety_gate), and returns one AgentResponse shape regardless of which
combination of inputs was given. safety_gate is the last node on every path
through the graph, so every response is guaranteed to carry a disclaimer —
this is what "code-enforced, non-bypassable" means in practice: it's not a
prompt instruction the LLM could ignore, it's a node that always runs.
"""

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class Citation(BaseModel):
    title: str
    url: str


class AgentResponse(BaseModel):
    """The one and only shape the agent returns."""

    # vision results — None if no image was submitted
    mask_png_base64: Optional[str] = None
    risk_score: Optional[float] = None
    risk_band: Optional[Literal["low", "moderate", "high"]] = None

    # RAG results — empty if refused or no question was asked
    answer: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)

    # safety gate output — always populated, on every path
    disclaimer: str
    refused: bool = False
    escalate: bool = False
    escalation_reason: Optional[str] = None


class AgentState(TypedDict, total=False):
    """Internal state threaded through the LangGraph nodes."""

    image_bytes: Optional[bytes]
    question: Optional[str]

    in_scope: bool

    mask_png_base64: Optional[str]
    risk_score: Optional[float]
    risk_band: Optional[str]
    raw_logit: Optional[float]

    retrieved_chunks: list  # rag.retriever.RetrievedChunk objects
    composed_answer: Optional[str]
    citations: list[dict]

    response: AgentResponse