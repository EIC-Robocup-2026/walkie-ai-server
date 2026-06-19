"""Tests for the unified Image API — /image/process and /image/embed-text.

Live integration tests: they POST to a running walkie-agent-v2 server (see
``--base-url``). The image is uploaded once; the JSON ``spec`` form field selects
which of the six tasks to run. Only requested keys appear in ``data``.
"""

import json

import requests


def _process(base_url, image_bytes, spec, return_status=False):
    resp = requests.post(
        f"{base_url}/image/process",
        files={"image": ("test.png", image_bytes, "image/png")},
        data={"spec": json.dumps(spec)},
    )
    if return_status:
        return resp
    body = resp.json()
    assert resp.status_code == 200, body
    assert body["success"] is True, body
    return body["data"]


# ---------------------------------------------------------------------------
# /image/process — validation
# ---------------------------------------------------------------------------

def test_process_missing_image(base_url):
    resp = requests.post(f"{base_url}/image/process", data={"spec": "{}"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "image" in body["error"].lower()


def test_process_invalid_image_bytes(base_url):
    resp = requests.post(
        f"{base_url}/image/process",
        files={"image": ("bad.png", b"not-an-image", "image/png")},
        data={"spec": '{"caption": true}'},
    )
    assert resp.status_code == 400
    assert resp.json()["success"] is False


def test_process_invalid_spec_json(base_url, sample_image_bytes):
    resp = requests.post(
        f"{base_url}/image/process",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"spec": "{not json"},
    )
    assert resp.status_code == 400
    assert resp.json()["success"] is False


def test_process_empty_spec_returns_empty_data(base_url, sample_image_bytes):
    data = _process(base_url, sample_image_bytes, {})
    assert data == {}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_detection_shape(base_url, sample_image_bytes):
    data = _process(base_url, sample_image_bytes, {"detection": True})
    assert isinstance(data["detection"], list)
    for det in data["detection"]:
        assert len(det["bbox"]) == 4
        assert "class_name" in det and "confidence" in det
        assert "area_ratio" in det and "class_id" in det
        assert "mask_b64" in det
        assert det["mask_b64"] is None  # no return_mask -> null


def test_detection_return_mask_and_prompts_accepted(base_url, sample_image_bytes):
    data = _process(
        base_url, sample_image_bytes,
        {"detection": {"return_mask": True, "prompts": ["cup", "bottle"]}},
    )
    for det in data["detection"]:
        assert det["mask_b64"] is None or isinstance(det["mask_b64"], str)


# ---------------------------------------------------------------------------
# Fused per-detection caption + embed
# ---------------------------------------------------------------------------

def test_per_detection_attaches_caption_and_embedding(base_url, sample_image_bytes):
    data = _process(
        base_url, sample_image_bytes,
        {
            "detection": {"return_mask": True},
            "per_detection": {"caption": True, "embed": True},
        },
    )
    for det in data["detection"]:
        # eligible detections carry the fused fields
        assert "caption" in det
        assert isinstance(det["embedding"], list)
        assert det["embedding_dim"] == len(det["embedding"])


# ---------------------------------------------------------------------------
# Whole-frame tasks
# ---------------------------------------------------------------------------

def test_caption(base_url, sample_image_bytes):
    data = _process(base_url, sample_image_bytes, {"caption": True})
    assert isinstance(data["caption"], str)


def test_caption_with_prompt(base_url, sample_image_bytes):
    data = _process(base_url, sample_image_bytes, {"caption": {"prompt": "What color?"}})
    assert isinstance(data["caption"], str)


def test_pose(base_url, sample_image_bytes):
    data = _process(base_url, sample_image_bytes, {"pose": True})
    assert isinstance(data["pose"], list)
    for p in data["pose"]:
        assert len(p["bbox"]) == 4
        assert "confidence" in p and isinstance(p["keypoints"], list)


def test_embed(base_url, sample_image_bytes):
    data = _process(base_url, sample_image_bytes, {"embed": True})
    assert isinstance(data["embed"]["embedding"], list)
    assert data["embed"]["dim"] == len(data["embed"]["embedding"])


def test_face(base_url, sample_image_bytes):
    data = _process(base_url, sample_image_bytes, {"face": True})
    assert isinstance(data["face"], list)
    for f in data["face"]:
        assert len(f["bbox_xyxy"]) == 4
        assert isinstance(f["embedding"], list) and "det_score" in f


def test_appearance(base_url, sample_image_bytes):
    data = _process(base_url, sample_image_bytes, {"appearance": True})
    assert isinstance(data["appearance"]["embedding"], list)


# ---------------------------------------------------------------------------
# Multi-task: one upload, several tasks
# ---------------------------------------------------------------------------

def test_multi_task_returns_all_requested_keys(base_url, sample_image_bytes):
    data = _process(base_url, sample_image_bytes, {"detection": True, "caption": True, "pose": True})
    assert set(data) == {"detection", "caption", "pose"}


# ---------------------------------------------------------------------------
# /image/embed-text
# ---------------------------------------------------------------------------

def test_embed_text_missing_text(base_url):
    resp = requests.post(f"{base_url}/image/embed-text", json={})
    assert resp.status_code == 400
    assert resp.json()["success"] is False


def test_embed_text_happy_path(base_url):
    body = requests.post(f"{base_url}/image/embed-text", json={"text": "a mug"}).json()
    assert body["success"] is True
    assert body["data"]["dim"] == len(body["data"]["embedding"])
