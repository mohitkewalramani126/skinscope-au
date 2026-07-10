"""
Lightweight, torch-free query/corpus embedder for SkinScope AU's RAG layer.

Day 14 rewrite: the original Day 10 retriever used sentence-transformers,
which pulls in a full PyTorch install (typically several hundred MB of
installed packages) just to embed 20 short corpus chunks and the occasional
query. That's fine on a dev machine, but it made the app too heavy to fit a
free-tier serverless host's bundle-size and memory limits (see Day 14 notes
in rag/retriever.py). This module replaces it with the same underlying
model -- sentence-transformers/all-MiniLM-L6-v2 -- run directly via
onnxruntime, with tokenization via the `tokenizers` library. Neither
dependency needs torch.

Faithfulness to the original model matters here: this reimplements the
model's own documented pipeline (see modules.json / 1_Pooling/config.json on
the model's Hugging Face page), not a guess --
  0_Transformer -> 1_Pooling (mean pooling over token embeddings, masked by
  attention_mask) -> 2_Normalize (L2 normalize).
Cosine similarity is invariant to whether inputs are pre-normalized, so the
L2-normalize step doesn't change retrieval *ranking* -- it's included anyway
to keep raw embedding values comparable to the original model's output
(e.g. if anything else ever wants a plain dot product instead of computed
cosine distance).

Model files (~90MB) come from
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main --
the plain fp32 `onnx/model.onnx`, not one of the quantized variants, to
avoid introducing any numerical drift from the embeddings the Day 10/11
out-of-scope threshold (0.65) was tuned against. If bundle size ever becomes
a problem, an int8-quantized variant (~23MB) is available, but would need
the threshold re-swept against evaluation/golden_qa.json before trusting it.
"""

import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

RAG_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(RAG_DIR, "embedding_model", "model.onnx")
TOKENIZER_PATH = os.path.join(RAG_DIR, "embedding_model", "tokenizer.json")
MAX_SEQ_LENGTH = 256  # matches sentence-transformers/all-MiniLM-L6-v2's default

_session: ort.InferenceSession | None = None
_tokenizer: Tokenizer | None = None


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def _get_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        _tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        _tokenizer.enable_padding(length=None)  # pad to longest in batch, not a fixed length
    return _tokenizer


def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Masked mean pooling over the token dimension -- padding tokens must not
    dilute the average. Matches sentence-transformers' Pooling module
    (pooling_mode_mean_tokens=True, see 1_Pooling/config.json)."""
    mask = attention_mask[:, :, None].astype(np.float32)  # (batch, seq_len, 1)
    summed = (token_embeddings * mask).sum(axis=1)  # (batch, hidden)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)  # avoid div-by-zero
    return summed / counts


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts, returning an (n, 384) float32 array of
    L2-normalized sentence embeddings -- same shape/space as the original
    sentence-transformers model's .encode()."""
    tokenizer = _get_tokenizer()
    encodings = tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    session = _get_session()
    input_names = {i.name for i in session.get_inputs()}
    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in input_names:
        onnx_inputs["token_type_ids"] = token_type_ids

    outputs = session.run(None, onnx_inputs)
    token_embeddings = outputs[0]  # (batch, seq_len, 384) -- last_hidden_state

    pooled = _mean_pool(token_embeddings, attention_mask)
    return _l2_normalize(pooled).astype(np.float32)


if __name__ == "__main__":
    vecs = embed(["What does the ABCDE rule mean?", "What's the weather like today?"])
    print("Shape:", vecs.shape)
    print("Norms (should be ~1.0):", np.linalg.norm(vecs, axis=1))
