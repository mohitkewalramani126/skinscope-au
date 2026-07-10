"""
One-time (re-runnable) setup script: downloads the ONNX + tokenizer files
rag/embedder.py needs, from sentence-transformers/all-MiniLM-L6-v2's own HF
Hub repo (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

Why this is a separate download rather than a pip package: the fp32
onnx/model.onnx file is ~90MB -- too large to bundle inside a pip package,
and not needed at all if you're not using this repo's RAG feature. Run this
once locally (or in CI before deploy) whenever rag/embedding_model/ is empty.

Run: python rag/download_embedding_model.py
"""

import os

from huggingface_hub import hf_hub_download

REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
RAG_DIR = os.path.dirname(__file__)
DEST_DIR = os.path.join(RAG_DIR, "embedding_model")


def main() -> None:
    os.makedirs(DEST_DIR, exist_ok=True)

    for filename, subpath in [("onnx/model.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")]:
        print(f"Downloading {filename}...")
        path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=DEST_DIR)
        # hf_hub_download preserves the "onnx/" subfolder; move model.onnx up
        # to rag/embedding_model/model.onnx to match rag/embedder.py's MODEL_PATH.
        target = os.path.join(DEST_DIR, subpath)
        if os.path.abspath(path) != os.path.abspath(target):
            os.replace(path, target)
        print(f"  -> {target} ({os.path.getsize(target) / 1e6:.1f} MB)")

    # clean up the now-empty onnx/ subfolder hf_hub_download created
    onnx_subdir = os.path.join(DEST_DIR, "onnx")
    if os.path.isdir(onnx_subdir) and not os.listdir(onnx_subdir):
        os.rmdir(onnx_subdir)

    print(f"\nDone. Model files are in {DEST_DIR}")


if __name__ == "__main__":
    main()
