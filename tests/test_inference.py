import base64


import pytest
from PIL import Image

from vision.inference import AnalysisResult, analyze_image


def _fake_jpeg_bytes(width=300, height=200, color=(200, 150, 130)):
    """Build a real, valid, tiny JPEG in memory — not a lesion, just something
    that decodes correctly, to test the pipeline's plumbing without depending
    on external test fixture files."""
    img = Image.new("RGB", (width, height), color=color)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_analyze_image_returns_valid_schema_on_normal_input():
    image_bytes = _fake_jpeg_bytes()
    result = analyze_image(image_bytes)

    assert isinstance(result, AnalysisResult)
    assert 0.0 <= result.risk_score <= 1.0
    assert result.risk_band in {"low", "moderate", "high"}
    assert isinstance(result.raw_logit, float)

    # mask must be valid, decodable PNG bytes
    mask_bytes = base64.b64decode(result.mask_png_base64)
    assert mask_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic number


def test_analyze_image_tiny_input():
    # 1x1 pixel image — pipeline resizes internally, so this should not crash
    image_bytes = _fake_jpeg_bytes(width=1, height=1)
    result = analyze_image(image_bytes)
    assert 0.0 <= result.risk_score <= 1.0


def test_analyze_image_large_input():
    # closer to a real phone photo's resolution
    image_bytes = _fake_jpeg_bytes(width=4032, height=3024)
    result = analyze_image(image_bytes)
    assert 0.0 <= result.risk_score <= 1.0


def test_analyze_image_rejects_invalid_bytes():
    garbage_bytes = b"this is not an image, just some random text bytes"
    with pytest.raises(ValueError):
        analyze_image(garbage_bytes)


def test_analyze_image_rejects_empty_bytes():
    with pytest.raises(ValueError):
        analyze_image(b"")