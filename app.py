from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from agent.graph import run_agent
from agent.schemas import AgentResponse
from vision.inference import AnalysisResult, analyze_image

app = FastAPI(title="SkinScope AU")


def health() -> dict:
    """Basic health check for the SkinScope service."""
    return {"status": "ok", "service": "skinscope-au"}


@app.get("/health")
def health_endpoint() -> dict:
    return health()


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)) -> AnalysisResult:
    """Day 9 endpoint: vision-only (mask + calibrated score), kept for backward
    compatibility with tests/test_inference.py. The custom frontend uses
    /api/analyze below instead, which also covers grounded Q&A + safety gate."""
    image_bytes = await file.read()
    try:
        return analyze_image(image_bytes)
    except ValueError as e:
        # bad/corrupt image input — a client error, not a server error
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze", response_model=AgentResponse)
async def api_analyze(
    file: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None),
    sensitivity_mode: str = Form("standard"),
) -> AgentResponse:
    """Day 12 endpoint: the full LangGraph agent. Image and question are both
    optional, but at least one is required -- this is what the custom
    frontend calls.

    sensitivity_mode: "standard" (escalate only on "high", 50.7% sensitivity /
    95.1% specificity) or "high" (also escalate on "moderate", 76.7%
    sensitivity / 86.1% specificity) -- both measured on the real Day 7 test
    set, see docs/model_card.md."""
    image_bytes = await file.read() if file is not None else None
    question_clean = question.strip() if question else None

    if sensitivity_mode not in ("standard", "high"):
        raise HTTPException(status_code=400, detail="sensitivity_mode must be 'standard' or 'high'.")

    if not image_bytes and not question_clean:
        raise HTTPException(status_code=400, detail="Provide an image, a question, or both.")

    try:
        return run_agent({
            "image_bytes": image_bytes,
            "question": question_clean,
            "sensitivity_mode": sensitivity_mode,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Must be mounted last -- it catches any request path not matched by a route
# defined above, and (with html=True) serves frontend/index.html at "/".
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    print(health())