"""Tests for the Object Detection API — /object-detection/*"""

import requests
import pytest


# ---------------------------------------------------------------------------
# /object-detection/providers
# ---------------------------------------------------------------------------

def test_object_detection_providers_returns_200(base_url):
    resp = requests.get(f"{base_url}/object-detection/providers")
    assert resp.status_code == 200


def test_object_detection_providers_success_shape(base_url):
    body = requests.get(f"{base_url}/object-detection/providers").json()
    assert body["success"] is True
    assert isinstance(body["data"], list)
    assert len(body["data"]) > 0


def test_object_detection_providers_includes_yoloe(base_url):
    body = requests.get(f"{base_url}/object-detection/providers").json()
    assert "yoloe" in body["data"]


# ---------------------------------------------------------------------------
# /object-detection/detect  — validation errors
# ---------------------------------------------------------------------------

def test_object_detection_detect_missing_image(base_url):
    resp = requests.post(f"{base_url}/object-detection/detect")
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "image" in body["error"].lower()


def test_object_detection_detect_invalid_image_bytes(base_url):
    resp = requests.post(
        f"{base_url}/object-detection/detect",
        files={"image": ("bad.png", b"not-an-image", "image/png")},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False


# ---------------------------------------------------------------------------
# /object-detection/detect  — happy path
# ---------------------------------------------------------------------------

def test_object_detection_detect_returns_list(base_url, sample_image_bytes):
    resp = requests.post(
        f"{base_url}/object-detection/detect",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert isinstance(body["data"], list)


def test_object_detection_detect_detection_shape(base_url, sample_image_bytes):
    body = requests.post(
        f"{base_url}/object-detection/detect",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    ).json()
    assert body["success"] is True
    for det in body["data"]:
        assert "bbox" in det
        assert "class_name" in det
        assert "confidence" in det
        assert "area_ratio" in det
        assert "class_id" in det
        assert "mask_b64" in det
        assert len(det["bbox"]) == 4
        assert 0.0 <= det["confidence"] <= 1.0
        assert 0.0 <= det["area_ratio"] <= 1.0


def test_object_detection_detect_mask_null_by_default(base_url, sample_image_bytes):
    """Without return_mask, each detection's mask_b64 is null."""
    body = requests.post(
        f"{base_url}/object-detection/detect",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    ).json()
    assert body["success"] is True
    for det in body["data"]:
        assert det["mask_b64"] is None


def test_object_detection_detect_return_mask_flag_accepted(base_url, sample_image_bytes):
    """return_mask=true is accepted; mask_b64 is null (default detect model) or a
    base64 string (segmentation-capable providers)."""
    resp = requests.post(
        f"{base_url}/object-detection/detect",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"return_mask": "true"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    for det in body["data"]:
        assert det["mask_b64"] is None or isinstance(det["mask_b64"], str)


def test_object_detection_detect_prompts_field_accepted(base_url, sample_image_bytes):
    """A comma-separated prompts field is accepted (no-op for YOLO)."""
    resp = requests.post(
        f"{base_url}/object-detection/detect",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"prompts": "cup,bottle", "return_mask": "true"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
