"""
Day 13 integration tests: exercise the real, wired-together pipeline end to
end (FastAPI -> agent.graph.run_agent -> check_scope/segment/classify/
retrieve/compose/safety_gate -> vision.inference) via the actual HTTP layer,
rather than calling node functions directly as tests/test_agent.py does.

Uses a real fixture image (test_images/IMG_2084.jpg) so segmentation/
classification actually run through the real ONNX models -- this is the one
place in the suite that proves the full wiring works, not just each node in
isolation.

GROQ_API_KEY is deliberately unset/cleared here so compose_node takes its
documented no-key fallback (plain extractive concatenation) -- this keeps the
test deterministic, free, and independent of network access, consistent with
the same tradeoff documented in tests/test_agent.py for compose_node.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app

FIXTURE_IMAGE = Path(__file__).parent.parent / "test_images" / "IMG_2084.jpg"
_MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_AVAILABLE = (
    (_MODELS_DIR / "classifier_model.onnx").exists()
    and (_MODELS_DIR / "segmentation_model.onnx").exists()
)
# models/*.onnx is gitignored (86MB + 16MB — too big for a plain git repo
# without LFS), so it won't exist on a fresh CI checkout until Day 14 sorts
# out model hosting for deployment. Tests that need real inference skip
# rather than fail in that environment -- this is a documented gap, not a
# silently-passing test.
requires_models = pytest.mark.skipif(
    not MODELS_AVAILABLE,
    reason="models/*.onnx not present (gitignored, not yet hosted for CI -- see Day 14 deploy notes)",
)


@pytest.fixture(autouse=True)
def _no_groq_key(monkeypatch):
    """Force the deterministic no-key fallback path for every test in this file."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


@pytest.fixture
def client():
    return TestClient(app)


@requires_models
def test_end_to_end_image_only(client):
    with open(FIXTURE_IMAGE, "rb") as f:
        resp = client.post("/api/analyze", files={"file": ("lesion.jpg", f, "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()

    # vision fields must be populated by the real models
    assert body["mask_png_base64"]
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_band"] in {"low", "moderate", "high"}

    # no question was asked, so no answer/citations, and never refused
    assert body["answer"] is None
    assert body["citations"] == []
    assert body["refused"] is False

    # the non-bypassable safety gate output must always be present
    assert body["disclaimer"]
    assert body["sensitivity_mode"] == "standard"  # default when not sent


@requires_models
def test_end_to_end_image_plus_in_scope_question(client):
    with open(FIXTURE_IMAGE, "rb") as f:
        resp = client.post(
            "/api/analyze",
            files={"file": ("lesion.jpg", f, "image/jpeg")},
            data={"question": "What does the ABCDE rule mean for checking moles?"},
        )
    assert resp.status_code == 200
    body = resp.json()

    # vision fields still populated
    assert body["mask_png_base64"]
    assert body["risk_band"] in {"low", "moderate", "high"}

    # in-scope question -> answered, not refused, with real citations
    assert body["refused"] is False
    assert body["answer"]
    assert len(body["citations"]) > 0
    assert body["disclaimer"]


@requires_models
def test_end_to_end_image_plus_out_of_scope_question_still_scores_image(client):
    # the core Day-11 design decision, proven here through the real HTTP path:
    # an off-topic question must not suppress a legitimate image analysis.
    with open(FIXTURE_IMAGE, "rb") as f:
        resp = client.post(
            "/api/analyze",
            files={"file": ("lesion.jpg", f, "image/jpeg")},
            data={"question": "What's the weather like in Sydney today?"},
        )
    assert resp.status_code == 200
    body = resp.json()

    assert body["refused"] is True
    assert body["citations"] == []
    # vision results survive the refusal
    assert body["mask_png_base64"]
    assert body["risk_band"] in {"low", "moderate", "high"}
    assert body["disclaimer"]


@requires_models
def test_end_to_end_high_sensitivity_mode_threaded_through(client):
    with open(FIXTURE_IMAGE, "rb") as f:
        resp = client.post(
            "/api/analyze",
            files={"file": ("lesion.jpg", f, "image/jpeg")},
            data={"sensitivity_mode": "high"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sensitivity_mode"] == "high"
    # escalate must be consistent with the band actually returned
    if body["risk_band"] in ("moderate", "high"):
        assert body["escalate"] is True
    else:
        assert body["escalate"] is False


def test_end_to_end_question_only_no_image(client):
    resp = client.post("/api/analyze", data={"question": "How often should I check my skin?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mask_png_base64"] is None
    assert body["risk_score"] is None
    assert body["risk_band"] is None
    assert body["answer"]
    assert body["disclaimer"]


def test_end_to_end_rejects_empty_request(client):
    resp = client.post("/api/analyze", data={})
    assert resp.status_code == 400


def test_end_to_end_rejects_invalid_sensitivity_mode(client):
    with open(FIXTURE_IMAGE, "rb") as f:
        resp = client.post(
            "/api/analyze",
            files={"file": ("lesion.jpg", f, "image/jpeg")},
            data={"sensitivity_mode": "extreme"},
        )
    assert resp.status_code == 400


@requires_models
def test_backward_compatible_vision_only_endpoint_still_works(client):
    # Day 9's /analyze endpoint, kept alive for tests/test_inference.py-style
    # callers and any external integration built against it before Day 12.
    with open(FIXTURE_IMAGE, "rb") as f:
        resp = client.post("/analyze", files={"file": ("lesion.jpg", f, "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["mask_png_base64"]
