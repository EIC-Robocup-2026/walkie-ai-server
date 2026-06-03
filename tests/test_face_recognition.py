"""Tests for the Face Recognition API — /face-recognition/*

Integration tests against a running server (like the other test_*.py here).
The happy-path embedding assertions need a real face in the frame, so they are
skipped unless ``FACE_TEST_IMAGE`` points at a JPEG/PNG with one clear face.
Set ``FACE_TEST_IMAGE_2`` to a second photo of the *same* person and
``FACE_TEST_IMAGE_OTHER`` to a *different* person to exercise the cosine margin.
"""

import math
import os

import pytest
import requests


def _embed(base_url, img_bytes, name="face.jpg"):
    return requests.post(
        f"{base_url}/face-recognition/embed",
        files={"image": (name, img_bytes, "image/jpeg")},
    )


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _cosine_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return 1.0 - dot  # vectors are L2-normalized server-side


# ---------------------------------------------------------------------------
# /face-recognition/providers + /info
# ---------------------------------------------------------------------------

def test_face_recognition_providers_success_shape(base_url):
    body = requests.get(f"{base_url}/face-recognition/providers").json()
    assert body["success"] is True
    assert isinstance(body["data"], list)
    assert "insightface" in body["data"]


def test_face_recognition_info_shape(base_url):
    body = requests.get(f"{base_url}/face-recognition/info").json()
    assert body["success"] is True
    assert isinstance(body["data"]["model_name"], str)
    assert isinstance(body["data"]["dim"], int)
    assert body["data"]["dim"] > 0


# ---------------------------------------------------------------------------
# /face-recognition/embed — validation errors
# ---------------------------------------------------------------------------

def test_face_recognition_embed_missing_image(base_url):
    resp = requests.post(f"{base_url}/face-recognition/embed")
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "image" in body["error"].lower()


def test_face_recognition_embed_invalid_image_bytes(base_url):
    resp = _embed(base_url, b"not-an-image", name="bad.jpg")
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False


# ---------------------------------------------------------------------------
# /face-recognition/embed — no face
# ---------------------------------------------------------------------------

def test_face_recognition_embed_no_face_returns_empty(base_url, sample_image_bytes):
    # A flat blue 100x100 image has no face → success with empty data.
    resp = _embed(base_url, sample_image_bytes, name="blank.png")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"] == []


# ---------------------------------------------------------------------------
# /face-recognition/embed — happy path (needs a real face image)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def face_image_bytes():
    path = os.getenv("FACE_TEST_IMAGE")
    if not path or not os.path.exists(path):
        pytest.skip("set FACE_TEST_IMAGE to a photo with one clear face")
    return _read(path)


def test_face_recognition_embed_single_face_shape(base_url, face_image_bytes):
    info = requests.get(f"{base_url}/face-recognition/info").json()["data"]
    body = _embed(base_url, face_image_bytes).json()
    assert body["success"] is True
    assert len(body["data"]) >= 1

    face = body["data"][0]
    assert len(face["bbox_xyxy"]) == 4
    x1, y1, x2, y2 = face["bbox_xyxy"]
    assert x2 > x1 and y2 > y1
    assert 0.0 <= face["det_score"] <= 1.0

    emb = face["embedding"]
    assert len(emb) == info["dim"]
    norm = math.sqrt(sum(v * v for v in emb))
    assert abs(norm - 1.0) < 1e-3  # L2-normalized contract


def test_face_recognition_same_vs_different_cosine_margin(base_url):
    same = os.getenv("FACE_TEST_IMAGE")
    same2 = os.getenv("FACE_TEST_IMAGE_2")
    other = os.getenv("FACE_TEST_IMAGE_OTHER")
    if not (same and same2 and other):
        pytest.skip("set FACE_TEST_IMAGE, FACE_TEST_IMAGE_2, FACE_TEST_IMAGE_OTHER")

    def first_embedding(path):
        data = _embed(base_url, _read(path)).json()["data"]
        assert data, f"no face found in {path}"
        # Largest bbox = the enrolled subject.
        data.sort(key=lambda f: (f["bbox_xyxy"][2] - f["bbox_xyxy"][0])
                  * (f["bbox_xyxy"][3] - f["bbox_xyxy"][1]), reverse=True)
        return data[0]["embedding"]

    e_same = first_embedding(same)
    e_same2 = first_embedding(same2)
    e_other = first_embedding(other)

    d_same = _cosine_distance(e_same, e_same2)
    d_other = _cosine_distance(e_same, e_other)
    assert d_same < d_other, f"same={d_same:.3f} not < different={d_other:.3f}"
