"""Shared utilities for the Flask API."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from flask import jsonify
from PIL import Image


def success(data: Any):
    """Return a standard success JSON response."""
    return jsonify({"success": True, "data": data})


def error(message: str, status: int = 400):
    """Return a standard error JSON response."""
    return jsonify({"success": False, "error": message}), status


def image_from_request_file(file) -> Image.Image:
    """Decode a werkzeug FileStorage into a PIL RGB Image."""
    return Image.open(file.stream).convert("RGB")


def image_from_bytes(data: bytes) -> Image.Image:
    """Decode raw bytes into a PIL RGB Image."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def crop_pil(img: Image.Image, bbox, margin: int = 20) -> Image.Image:
    """Crop *img* to ``bbox`` (x1, y1, x2, y2) with a clamped pixel *margin*.

    The margin gives the captioning / embedding models some surrounding context
    (ConceptGraphs pads its feature crops by 20 px for the same reason). Mirrors
    the agent-side ``walkie_graphs`` crop so a fused crop matches the thumbnail
    the caller stores. Returns the full image if the box degenerates.
    """
    w, h = img.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = min(w, int(x2) + margin)
    y2 = min(h, int(y2) + margin)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def pil_to_b64(img: Image.Image | None, fmt: str = "PNG") -> str | None:
    """Encode a PIL Image to a base64 string, or return None."""
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def mask_to_b64(mask: Any) -> str | None:
    """Encode a numpy mask array to a base64 PNG string, or return None."""
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr * 255, mode="L")
    else:
        img = Image.fromarray(arr)
    return pil_to_b64(img)


def extract_provider_config(source: dict) -> tuple[str | None, dict]:
    """Extract the ``provider`` key from a flat dict and return remaining keys as config."""
    data = dict(source)
    provider = data.pop("provider", None)
    return provider, data
