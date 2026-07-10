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
    sensitivity_mode: Literal["standard", "high"] = "standard"


class AgentState(TypedDict, total=False):
    """Internal state threaded through the LangGraph nodes."""

    image_bytes: Optional[bytes]
    question: Optional[str]
    # "standard": escalate only on the "high" band (score >= 0.7918) --
    #   50.7% sensitivity / 95.1% specificity, measured Day 7.
    # "high": also escalate on "moderate" (score >= 0.5) -- 76.7% sensitivity /
    #   86.1% specificity, same test set. Calibrated 0.5 == raw-probability 0.5
    #   exactly (temperature scaling doesn't move the 0.5 point), so this is
    #   not a new number, it's the already-measured threshold-0.5 operating
    #   point from docs/model_card.md.
    sensitivity_mode: Literal["standard", "high"]

    in_scope: bool

    mask_png_base64: Optional[str]
    risk_score: Optional[float]
    risk_band: Optional[str]
    raw_logit: Optional[float]

    retrieved_chunks: list  # rag.retriever.RetrievedChunk objects
    composed_answer: Optional[str]
    citations: list[dict]

    response: AgentResponse