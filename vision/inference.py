"""
Inference pipeline for SkinScope AU.

Two independent ONNX models, run in parallel over the SAME full (uncropped) input
image:
  - segmentation_model.onnx: DeepLabV3+/resnet34, outputs a lesion mask
  - classifier_model.onnx:   EfficientNet-B0, outputs a malignancy risk logit

IMPORTANT: the classifier was trained on FULL images, not lesion crops (a deliberate
Day-6 scope decision — see docs/model_card.md). The segmentation output is NOT used to
crop the classifier's input. Preprocessing below must exactly match training
(vision preprocessing parity is the #1 production bug for ML services).
"""

import os

import cv2
import numpy as np
import onnxruntime as ort
import base64
from typing import Literal
from pydantic import BaseModel

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SEGMENTATION_MODEL_PATH = os.path.join(MODELS_DIR, "segmentation_model.onnx")
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "classifier_model.onnx")

# must match training exactly — see notebooks/model_training_and_evaluation.ipynb
SEGMENTATION_IMG_SIZE = 256
CLASSIFIER_IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# learned on the classifier's validation set (Day 7) — never touch this without re-fitting
CLASSIFIER_TEMPERATURE = 1.674

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid — avoids overflow warnings on extreme logits."""
    x_clipped = np.clip(x, -500, 500)
    return np.where(x_clipped >= 0, 1 / (1 + np.exp(-x_clipped)), np.exp(x_clipped) / (1 + np.exp(x_clipped)))

_segmentation_session: ort.InferenceSession | None = None
_classifier_session: ort.InferenceSession | None = None


def get_segmentation_session() -> ort.InferenceSession:
    global _segmentation_session
    if _segmentation_session is None:
        _segmentation_session = ort.InferenceSession(SEGMENTATION_MODEL_PATH)
    return _segmentation_session


def get_classifier_session() -> ort.InferenceSession:
    global _classifier_session
    if _classifier_session is None:
        _classifier_session = ort.InferenceSession(CLASSIFIER_MODEL_PATH)
    return _classifier_session


# Day 14 (first attempt): tried dropping cv2 entirely in favor of PIL, to save the
# ~152MB opencv-python-headless install. Reverted — PIL's Image.BILINEAR and cv2's
# INTER_LINEAR are NOT equivalent at large downscale ratios (these source photos are
# ~4000px, resized to 224-256px, an ~18x reduction), and swapping produced a real,
# measured regression: risk_score moved by 2-3x and one test image's risk_band
# flipped from "low" to "moderate". Verified via before/after comparison on
# test_images/IMG_2084.jpg and IMG_2085.jpg — not floating-point noise.
#
# Kept instead: only `albumentations` itself was removed (its Resize/Normalize were
# just thin wrappers around cv2.resize + manual normalize anyway), which still drops
# the ~103MB transitive scipy dependency with zero behavior change, since cv2 itself
# is untouched below.
def _resize_and_normalize(image_rgb: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize to size x size (cv2.INTER_LINEAR, matching training and
    albumentations.Resize's default), then ImageNet normalize (matches
    albumentations.Normalize's default: divide by 255, then (x - mean) / std)."""
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def load_image_rgb(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes (upload) into an HxWx3 uint8 RGB array."""
    if not image_bytes:
        raise ValueError("Empty image data — no bytes to decode")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image — not a valid image file")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_for_segmentation(image_rgb: np.ndarray) -> np.ndarray:
    normalized = _resize_and_normalize(image_rgb, SEGMENTATION_IMG_SIZE)
    chw = normalized.transpose(2, 0, 1)
    return np.expand_dims(chw, axis=0).astype(np.float32)


def preprocess_for_classifier(image_rgb: np.ndarray) -> np.ndarray:
    normalized = _resize_and_normalize(image_rgb, CLASSIFIER_IMG_SIZE)
    chw = normalized.transpose(2, 0, 1)
    return np.expand_dims(chw, axis=0).astype(np.float32)



# risk bands anchored to actual measured operating points from Day 7's evaluation,
# not arbitrary round numbers: 0.5 is the naive midpoint, 0.7918 is the calibrated-probability
# threshold that achieved ~95% specificity on the test set (see docs/model_card.md)
LOW_RISK_MAX = 0.5
MODERATE_RISK_MAX = 0.7918


class AnalysisResult(BaseModel):
    mask_png_base64: str  # binary lesion mask, PNG-encoded, resized to the input image's dimensions
    risk_score: float  # calibrated probability of malignancy, in [0, 1]
    risk_band: Literal["low", "moderate", "high"]
    raw_logit: float  # uncalibrated model output, kept for debugging/traceability only


def _mask_to_png_base64(mask: np.ndarray) -> str:
    """mask: HxW array of 0/1 (or 0/255). Encodes as a PNG and returns base64 text."""
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    success, buffer = cv2.imencode(".png", mask_uint8)
    if not success:
        raise ValueError("Failed to encode mask as PNG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _risk_band(calibrated_score: float) -> str:
    if calibrated_score < LOW_RISK_MAX:
        return "low"
    if calibrated_score < MODERATE_RISK_MAX:
        return "moderate"
    return "high"


def analyze_image(image_bytes: bytes) -> AnalysisResult:
    image_rgb = load_image_rgb(image_bytes)
    original_h, original_w = image_rgb.shape[:2]

    # segmentation — full image in, mask out, resized back to the original image's dimensions
    seg_input = preprocess_for_segmentation(image_rgb)
    seg_logits = get_segmentation_session().run(None, {"input": seg_input})[0]
    seg_probs = _sigmoid(seg_logits[0, 0])  # replaces: 1 / (1 + np.exp(-seg_logits[0, 0]))
    mask_small = (seg_probs > 0.5).astype(np.uint8)
    # NEAREST, not bilinear — this is a binary 0/1 mask; any smoothing interpolation
    # would blur it into invalid in-between values.
    mask_full_size = cv2.resize(
        mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST
    )

    # classification — SAME full image, independent pass, never cropped by the mask above
    cls_input = preprocess_for_classifier(image_rgb)
    cls_logit = float(get_classifier_session().run(None, {"input": cls_input})[0][0][0])
    calibrated_score = float(_sigmoid(np.array(cls_logit / CLASSIFIER_TEMPERATURE)))  # replaces the manual 1/(1+np.exp(...)) line

    return AnalysisResult(
        mask_png_base64=_mask_to_png_base64(mask_full_size),
        risk_score=calibrated_score,
        risk_band=_risk_band(calibrated_score),
        raw_logit=cls_logit,
    )


if __name__ == "__main__":
    seg_session = get_segmentation_session()
    cls_session = get_classifier_session()

    dummy_seg_input = np.random.randn(1, 3, SEGMENTATION_IMG_SIZE, SEGMENTATION_IMG_SIZE).astype(np.float32)
    dummy_cls_input = np.random.randn(1, 3, CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE).astype(np.float32)

    seg_out = seg_session.run(None, {"input": dummy_seg_input})[0]
    cls_out = cls_session.run(None, {"input": dummy_cls_input})[0]

    print("segmentation output shape:", seg_out.shape)
    print("classifier output shape:", cls_out.shape)
    print("both ONNX sessions loaded and ran successfully")
