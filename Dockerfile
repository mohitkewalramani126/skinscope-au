# SkinScope AU -- Docker build for any container-based host (Render, Fly.io,
# a self-hosted VPS, etc.). NOT used by the current Vercel deployment, which
# uses Vercel's native Python runtime (root app.py as the entrypoint) instead
# of a Dockerfile -- kept here in case a container host is used later.

FROM python:3.11-slim

# libgl1 + libglib2.0-0: required by opencv-python-headless at import time
# (cv2 dlopen's libGL even in the "headless" build) -- without these the
# container crashes on the first `import cv2` inside vision/inference.py.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY agent/ ./agent/
COPY vision/ ./vision/
COPY rag/ ./rag/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY docs/ ./docs/

# 7860 matches Hugging Face Spaces' convention; harmless default for any
# other host, which will typically override PORT via its own env var.
ENV PORT=7860
EXPOSE 7860

# rag/embeddings.json and rag/embedding_model/ are committed to the repo
# (Day 14) -- no rebuild step needed at container start anymore.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
