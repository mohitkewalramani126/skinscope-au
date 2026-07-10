"""
Day 14: one-time (re-runnable) deploy script -- creates a Hugging Face Space
(Docker SDK) and uploads the whole project to it, including models/*.onnx.

Why this exists / why not plain git+git-lfs: models/*.onnx (102MB total) is
gitignored in the GitHub repo (see .gitignore) and was never committed there.
A Hugging Face Space is a *separate* git repository hosted by HF, so that's
fine -- the GitHub repo doesn't need to change at all. huggingface_hub's
upload_folder() handles large binary files (LFS) automatically, so there's
no need to install or configure git-lfs by hand.

This does NOT touch your GitHub repo or its .gitignore. It only pushes to
the HF Space repo, which is where the running deployment actually lives.

One-time setup before running this:
  1. Create a free account at https://huggingface.co/join (if you don't have one).
  2. Create a User Access Token: https://huggingface.co/settings/tokens
     (needs "Write" permission).
  3. pip install huggingface_hub
  4. huggingface-cli login   (paste the token when prompted)

Then run:
  python deploy_to_hf_space.py --username YOUR_HF_USERNAME

Re-running this script is safe -- it re-uploads any changed files to the
same Space (exist_ok=True), so it's also how you push updates later.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

# Files/folders never pushed to the Space:
#   - .git, __pycache__, .pytest_cache: local tooling artifacts, not app code
#   - tests/, notebooks/: dev-time only, not needed to run the deployed app
#   - test_images/: only used by tests/test_integration.py; the frontend has
#     its own copies in frontend/images/ for the live sample-photo buttons
#   - rag/chroma_db/: gitignored and rebuilt by the Dockerfile's CMD at
#     container start (see Dockerfile), so shipping a stale copy is pointless
#   - agent/traces.jsonl: gitignored local trace log, not app code
#   - .env: local secrets -- Space secrets are set separately in Settings,
#     never uploaded as a file (see the printed instructions below)
IGNORE_PATTERNS = [
    ".git*",
    "__pycache__/*",
    "**/__pycache__/*",
    ".pytest_cache/*",
    "tests/*",
    "notebooks/*",
    "test_images/*",
    "rag/chroma_db/*",
    "agent/traces.jsonl",
    ".env",
    "deploy_to_hf_space.py",  # this script itself doesn't need to ship
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--space-name", default="skinscope-au", help="Space name (repo_id will be username/space-name)")
    args = parser.parse_args()

    repo_id = f"{args.username}/{args.space_name}"
    api = HfApi()

    print(f"Creating (or reusing) Space: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    root = Path(__file__).parent
    print(f"Uploading {root} to {repo_id} (this includes models/*.onnx, ~102MB total)...")
    api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=IGNORE_PATTERNS,
    )

    print(f"""
Done. Space is building at: https://huggingface.co/spaces/{repo_id}

IMPORTANT -- before the app will actually work, set these as Space secrets
(Settings tab on the Space page, not as a committed .env file):
  - GROQ_API_KEY        (required -- compose_node falls back to extractive
                          answers without it, but Groq-phrased answers need it)
  - LANGFUSE_SECRET_KEY  (optional -- tracing no-ops cleanly without it)
  - LANGFUSE_PUBLIC_KEY  (optional)
  - LANGFUSE_BASE_URL    (optional)

The Space will take a few minutes to build the Docker image the first time.
Check the "Logs" tab on the Space page if it doesn't come up.
""")


if __name__ == "__main__":
    main()
