"""Tests for the Appearance (attire) re-ID API — /appearance/*

Integration tests against a running server (like the other test_*.py here).
Requires torchreid installed server-side (see README "Appearance re-ID").

Unlike /face-recognition/embed, /appearance/embed embeds **whatever image it
is given** — even a blank frame yields a (meaningless but valid) vector, so
the happy-path shape tests need no real photo. Set ``APPEARANCE_TEST_IMAGE``
+ ``APPEARANCE_TEST_IMAGE_2`` to two crops of the *same clothed person*
(different angles) and ``APPEARANCE_TEST_IMAGE_OTHER`` to a *different*
person in different clothes to exercise the cosine margin.
"""

import math
import os

import pytest
import requests


def _embed(base_url, img_bytes, name="person.jpg"):
    return requests.post(
        f"{base_url}/appearance/embed",
        files={"image": (name, img_bytes, "image/jpeg")},
    )


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _cosine_similarity(a, b):
    return sum(x * y for x, y in zip(a, b))  # vectors are L2-normalized server-side


# ---------------------------------------------------------------------------
# /appearance/providers + /info
# ---------------------------------------------------------------------------

def test_appearance_providers_success_shape(base_url):
    body = requests.get(f"{base_url}/appearance/providers").json()
    assert body["success"] is True
    assert isinstance(body["data"], list)
    assert "osnet" in body["data"]


def test_appearance_info_shape(base_url):
    body = requests.get(f"{base_url}/appearance/info").json()
    assert body["success"] is True
    assert isinstance(body["data"]["model_name"], str)
    assert isinstance(body["data"]["dim"], int)
    assert body["data"]["dim"] > 0


# ---------------------------------------------------------------------------
# /appearance/embed — validation errors
# ---------------------------------------------------------------------------

def test_appearance_embed_missing_image(base_url):
    resp = requests.post(f"{base_url}/appearance/embed")
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "image" in body["error"].lower()


def test_appearance_embed_invalid_image_bytes(base_url):
    resp = _embed(base_url, b"not-an-image", name="bad.jpg")
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False


# ---------------------------------------------------------------------------
# /appearance/embed — shape contract (any decodable image embeds)
# ---------------------------------------------------------------------------

def test_appearance_embed_returns_normalized_vector(base_url, sample_image_bytes):
    info = requests.get(f"{base_url}/appearance/info").json()["data"]
    resp = _embed(base_url, sample_image_bytes, name="blank.png")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True

    emb = body["data"]["embedding"]
    assert len(emb) == info["dim"]
    norm = math.sqrt(sum(v * v for v in emb))
    assert abs(norm - 1.0) < 1e-3  # L2-normalized contract


def test_appearance_embed_is_deterministic(base_url, sample_image_bytes):
    e1 = _embed(base_url, sample_image_bytes).json()["data"]["embedding"]
    e2 = _embed(base_url, sample_image_bytes).json()["data"]["embedding"]
    assert _cosine_similarity(e1, e2) > 0.999


# ---------------------------------------------------------------------------
# /appearance/embed — re-ID margin (needs real person crops)
# ---------------------------------------------------------------------------

def test_appearance_same_vs_different_cosine_margin(base_url):
    same = os.getenv("APPEARANCE_TEST_IMAGE")
    same2 = os.getenv("APPEARANCE_TEST_IMAGE_2")
    other = os.getenv("APPEARANCE_TEST_IMAGE_OTHER")
    if not (same and same2 and other):
        pytest.skip(
            "set APPEARANCE_TEST_IMAGE, APPEARANCE_TEST_IMAGE_2 (same clothed "
            "person, different angle), APPEARANCE_TEST_IMAGE_OTHER (different person)"
        )

    def embedding(path):
        body = _embed(base_url, _read(path)).json()
        assert body["success"] is True, f"embed failed for {path}"
        return body["data"]["embedding"]

    e_same = embedding(same)
    e_same2 = embedding(same2)
    e_other = embedding(other)

    s_same = _cosine_similarity(e_same, e_same2)
    s_other = _cosine_similarity(e_same, e_other)
    assert s_same > s_other, f"same={s_same:.3f} not > different={s_other:.3f}"
