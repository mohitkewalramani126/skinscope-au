

# SkinScope AU

A responsible skin-cancer *awareness* project — **not a diagnostic tool**. Upload a
lesion photo, ask a question, or both. The app returns a lesion segmentation mask, a
calibrated risk score, and a cited, grounded answer to skin-cancer questions — every
response passes through a single safety gate that attaches a disclaimer and decides
whether to escalate to "see a doctor."



## Screenshots

<img width="1906" height="880" alt="image" src="https://github.com/user-attachments/assets/14bb00e4-0039-492d-9db6-af439a5bf59b" />




<img width="1894" height="915" alt="image" src="https://github.com/user-attachments/assets/64cdad51-133d-40d9-91de-f762f6237822" />






## What it does

- **Segmentation** — DeepLabV3+ (ResNet-34 encoder) locates the lesion in the photo.
- **Risk classification** — EfficientNet-B0 outputs a calibrated malignancy risk score
  (low / moderate / high), trained on HAM10000.
- **Grounded Q&A** — a small, hand-curated, cited corpus (Cancer Council Australia,
  healthdirect, Melanoma Institute Australia) answers skin-cancer awareness questions
  via retrieval + Groq-generated phrasing. Off-topic questions are refused, not guessed at.
- **Safety gate** — every response (image, text, or both) passes through one final node
  that attaches a fixed medical disclaimer and decides whether to escalate to "see a
  doctor," based on the measured sensitivity/specificity trade-off at the current
  operating point.
- **Sensitivity mode** — a toggle between two measured operating points: *standard*
  (escalates only on "high" risk, 50.7% sensitivity / 95.1% specificity) or *high*
  (also escalates on "moderate," 76.7% sensitivity / 86.1% specificity). Neither is
  "the safe one" — the toggle exists because that trade-off is a judgment call, not
  something the model can decide on its own.

## Why the numbers matter more than the demo

This project treats "our model works" as a claim that needs evidence, not a given. The
real, measured numbers below are the actual point:

| | Value |
|---|---|
| Classifier test AUC | 0.9049 |
| Sensitivity @ ~95% specificity | **50.7%** — misses about half of malignant cases at this operating point |
| Sensitivity @ threshold 0.5 | 76.7% (specificity 86.1%) |
| Calibration (ECE), before → after temperature scaling | 0.1011 → 0.0842 |
| Segmentation test IoU / Dice | 0.8127 / 0.8820 |
| RAG scope-gate accuracy (27 golden questions) | 92.59% (25/27) |
| RAG citation coverage | 100% |
| Agent answer faithfulness (LLM-judge, human-corrected) | 80.0% → 90.0% after manual review |

Full detail, methodology, and honest limitations for each number live in:

- [`docs/model_card.md`](docs/model_card.md) — classifier training, calibration, and a
  fairness audit across skin-tone quartiles (sensitivity degrades from 55.6% in the
  lightest quartile to 41.9% in the darkest — a real, documented gap, not glossed over).
- [`evaluation/segmentation_report.md`](evaluation/segmentation_report.md) — segmentation
  training, qualitative failure analysis, and a caught-and-fixed train/val leakage bug.
- [`evaluation/rag_eval_report.md`](evaluation/rag_eval_report.md) — retrieval threshold
  tuning and the scope gate's known blind spot on terse, context-free questions.
- [`evaluation/agent_eval_report.md`](evaluation/agent_eval_report.md) — LLM-as-judge
  faithfulness evaluation of the full agent's generated answers, with every flagged
  answer manually reviewed rather than trusted blind.
- [`docs/data_sources.md`](docs/data_sources.md) — dataset provenance and licensing.

## Architecture

```
Image + question
      │
      ▼
 check_scope ──► retrieve ──► compose  (grounded Q&A, refuses off-topic)
      │
      ▼
 segment ──► classify         (mask + calibrated risk score)
      │
      ▼
 safety_gate  (single response constructor — disclaimer + escalation, always)
```

Built with **FastAPI** + **LangGraph**. Vision inference runs on **ONNX Runtime**
(exported from PyTorch, verified numerically equivalent — see the reports above). The
RAG layer embeds its 20-chunk corpus via a local ONNX Runtime + `tokenizers` pipeline
(no `sentence-transformers`/`torch` at runtime) and searches it with plain NumPy cosine
similarity — deliberately lightweight, since a full vector database adds nothing at
this corpus size. Answer phrasing uses Groq (`llama-3.1-8b-instant`).

## Running it locally

```bash
pip install -r requirements.txt -r requirements-dev.txt
export GROQ_API_KEY=your_key_here   # required for grounded Q&A phrasing
uvicorn app:app --reload
```

Then open `http://localhost:8000`. `GROQ_API_KEY` is required for the Q&A phrasing
step; vision-only requests (`/analyze`) work without it. Optional: set
`LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` for tracing (install `langfuse` from
`requirements-dev.txt` first — it's not in the production bundle, see
`requirements.txt`'s comments for why).

## Tests and CI

```bash
pytest -q --cov --cov-report=term-missing
ruff check .
```

GitHub Actions runs both on every push (`.github/workflows/ci.yml`). A consolidated
eval harness (`evaluation/run_all.py`) re-runs the RAG and agent evaluations above in
one command, and `evaluation/log_to_mlflow.py` / `evaluation/drift_report.py` cover
experiment tracking and data-drift monitoring (Evidently).

## Deployment

Deployed on Vercel's free tier. Getting the Python function bundle under Vercel's
500MB limit required real trade-offs, documented inline in `requirements.txt` and
`vision/inference.py`:

- `sentence-transformers` + `chromadb` (torch) → `onnxruntime` + `tokenizers` +
  NumPy cosine search, re-verified to produce identical retrieval accuracy.
- `albumentations` (pulls in `scipy`) → the same preprocessing via direct OpenCV
  calls. **`opencv-python-headless` itself was deliberately kept** — an earlier
  attempt to replace it with PIL changed the classifier's actual output (one test
  image's risk band flipped from "low" to "moderate"), which was rejected as an
  unacceptable trade-off for a safety-relevant score.
- Dev-only tooling (`pytest`, `ruff`, `evidently`, `mlflow`, `uvicorn[standard]`,
  `langfuse`) lives in `requirements-dev.txt`, not the deployed bundle.

## Known limitations (see the linked reports for full detail)

- The classifier misses roughly half of malignant cases at its high-specificity
  operating point — this is stated plainly, not buried in an appendix.
- The fairness audit uses a simplified lightness-only proxy for skin tone (HAM10000
  has no real skin-tone labels) and cannot make claims about performance on genuinely
  dark skin, due to the dataset's light-skin skew.
- The RAG scope gate has a known blind spot on very short, context-free questions
  ("Am I at risk?") that would need conversational context to resolve.
- This is an awareness tool. It does not diagnose skin cancer, and a "low risk" result
  does not rule it out. Always see a doctor or dermatologist for any new, changing, or
  concerning spot.
