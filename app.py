from fastapi import FastAPI, File, HTTPException, UploadFile

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
    image_bytes = await file.read()
    try:
        return analyze_image(image_bytes)
    except ValueError as e:
        # bad/corrupt image input — a client error, not a server error
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    print(health())