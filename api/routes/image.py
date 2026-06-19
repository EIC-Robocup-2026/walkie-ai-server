"""Unified image-processing blueprint — ``/image/*``.

One request, one image upload, any combination of the six vision tasks
(``detection``, ``caption``, ``pose``, ``embed``, ``face``, ``appearance``).
The request is multipart: an ``image`` file plus a JSON ``spec`` form field
selecting the tasks (and their per-task options). The handler runs only the
requested tasks against the shared model singletons in :mod:`api.models` and
returns a dict keyed by task — only requested keys are present.

A ``per_detection`` block fuses the heaviest pipeline: after ``detection``,
each detected crop is captioned and/or embedded server-side (no per-crop
network round-trip), and the results are attached onto each detection.

Spec shape::

    {
      "detection":  {"prompts": ["chair"], "return_mask": true},
      "caption":    {"prompt": "Describe the table"},   // or true
      "pose":       true,
      "embed":      true,
      "face":       true,
      "appearance": true,
      "per_detection": {                                 // requires "detection"
          "caption": {"prompt_template": "Describe the {class_name}.",
                      "classes": ["mug"]},               // or true
          "embed": true,
          "crop_margin_px": 20
      }
    }
"""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, request

from api.models import (
    get_appearance,
    get_caption,
    get_embedding,
    get_face,
    get_object_detection,
    get_pose,
)
from api.utils import (
    crop_pil,
    error,
    image_from_request_file,
    mask_to_b64,
    success,
)
from services import debug_viewer

bp = Blueprint("image", __name__, url_prefix="/image")


# ----------------------------------------------------------------------------
# Spec normalization
# ----------------------------------------------------------------------------

def _opts(val: Any) -> dict | None:
    """Normalize a task value to an options dict, or ``None`` when disabled.

    ``None``/``False`` → disabled; ``True`` → enabled with defaults; a dict is
    passed through; any other truthy scalar enables with defaults.
    """
    if val is None or val is False:
        return None
    if val is True:
        return {}
    if isinstance(val, dict):
        return val
    return {}


def _prompts(val: Any) -> list[str] | None:
    """Accept a list or comma-separated string of prompts; empty → None."""
    if val is None:
        return None
    if isinstance(val, str):
        val = val.split(",")
    cleaned = [str(p).strip() for p in val if str(p).strip()]
    return cleaned or None


# ----------------------------------------------------------------------------
# Serializers (shape-compatible with the old per-task routes)
# ----------------------------------------------------------------------------

def _serialize_detection(
    obj,
    caption: str | None = None,
    embedding: list[float] | None = None,
    embedding_dim: int | None = None,
) -> dict:
    out = {
        "bbox": list(obj.bbox),
        "area_ratio": obj.area_ratio,
        "class_id": obj.class_id,
        "class_name": obj.class_name,
        "confidence": obj.confidence,
        "mask_b64": mask_to_b64(obj.mask),
    }
    if caption is not None:
        out["caption"] = caption
    if embedding is not None:
        out["embedding"] = embedding
        out["embedding_dim"] = embedding_dim
    return out


def _serialize_pose(pose) -> dict:
    return {
        "bbox": list(pose.bbox),
        "confidence": pose.confidence,
        "keypoints": [
            {"index": kp.index, "name": kp.name, "x": kp.x, "y": kp.y,
             "confidence": kp.confidence}
            for kp in pose.keypoints
        ],
    }


def _serialize_face(face) -> dict:
    return {
        "bbox_xyxy": list(face.bbox_xyxy),
        "embedding": face.embedding,
        "det_score": face.det_score,
    }


# ----------------------------------------------------------------------------
# Fused per-detection caption + embed
# ----------------------------------------------------------------------------

def _run_per_detection(
    image, detections, opts: dict
) -> tuple[dict[int, str], dict[int, list[float]], int | None]:
    """Caption and/or embed each *eligible* detected crop server-side.

    Eligibility mirrors the gates the caller would otherwise apply before
    captioning/embedding: a detection is skipped if its class is in
    ``exclude_classes`` or its confidence is below ``min_confidence``. (Depth /
    point-count geometry gates live caller-side and can't be replicated here, so
    a few crops that the caller ultimately drops may still be processed.)

    Returns ``(captions_by_index, embeddings_by_index, embedding_dim)`` keyed by
    index into *detections* — only eligible indices appear.
    """
    margin = int(opts.get("crop_margin_px", 20))
    exclude = {c.lower() for c in (opts.get("exclude_classes") or [])}
    min_conf = float(opts.get("min_confidence", 0.0) or 0.0)
    eligible = [
        i for i, d in enumerate(detections)
        if (d.class_name or "").lower() not in exclude
        and float(d.confidence or 0.0) >= min_conf
    ]
    crops = {i: crop_pil(image, detections[i].bbox, margin) for i in eligible}

    captions: dict[int, str] = {}
    cap_opts = _opts(opts.get("caption"))
    if cap_opts is not None and eligible:
        classes = cap_opts.get("classes")
        classes_lower = {c.lower() for c in classes} if classes else None
        template = cap_opts.get("prompt_template") or cap_opts.get("prompt")
        idx = [
            i for i in eligible
            if classes_lower is None or (detections[i].class_name or "").lower() in classes_lower
        ]
        if idx:
            imgs = [crops[i] for i in idx]
            prompts = None
            if template:
                prompts = [
                    template.replace("{class_name}", detections[i].class_name or "object")
                    for i in idx
                ]
            out = get_caption().caption_batch(imgs, prompts=prompts)
            captions = {i: (c or "") for i, c in zip(idx, out)}

    embeddings: dict[int, list[float]] = {}
    embedding_dim: int | None = None
    if opts.get("embed") and eligible:
        emb = get_embedding()
        embedding_dim = emb.get_embedding_dim()
        embeddings = {i: emb.embed_image(crops[i]) for i in eligible}

    return captions, embeddings, embedding_dim


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@bp.post("/process")
def process():
    if "image" not in request.files:
        return error("Missing 'image' file in request")
    try:
        image = image_from_request_file(request.files["image"])
    except Exception as exc:
        return error(f"Invalid image: {exc}")

    raw_spec = request.form.get("spec")
    try:
        spec: dict = json.loads(raw_spec) if raw_spec else {}
    except Exception as exc:
        return error(f"Invalid spec JSON: {exc}")
    if not isinstance(spec, dict):
        return error("spec must be a JSON object")

    data: dict[str, Any] = {}
    try:
        # --- Detection (+ optional fused per-detection caption/embed) -------
        det_opts = _opts(spec.get("detection"))
        if det_opts is not None:
            detections = get_object_detection().detect(
                image,
                _prompts(det_opts.get("prompts")),
                return_mask=bool(det_opts.get("return_mask")),
            )
            debug_viewer.show_object_detection(image, detections)
            captions: dict[int, str] = {}
            embeddings: dict[int, list[float]] = {}
            emb_dim: int | None = None
            pd_opts = _opts(spec.get("per_detection"))
            if pd_opts is not None:
                captions, embeddings, emb_dim = _run_per_detection(
                    image, detections, pd_opts
                )
            data["detection"] = [
                _serialize_detection(
                    d,
                    caption=captions.get(i),
                    embedding=embeddings.get(i),
                    embedding_dim=emb_dim if i in embeddings else None,
                )
                for i, d in enumerate(detections)
            ]

        # --- Whole-frame caption -------------------------------------------
        cap_opts = _opts(spec.get("caption"))
        if cap_opts is not None:
            caption = get_caption().caption(image, prompt=cap_opts.get("prompt"))
            debug_viewer.show_image_caption(image, caption)
            data["caption"] = caption

        # --- Pose ----------------------------------------------------------
        if _opts(spec.get("pose")) is not None:
            poses = get_pose().estimate(image)
            debug_viewer.show_pose_estimation(image, poses)
            data["pose"] = [_serialize_pose(p) for p in poses]

        # --- Whole-frame image embed ---------------------------------------
        if _opts(spec.get("embed")) is not None:
            emb = get_embedding()
            data["embed"] = {
                "embedding": emb.embed_image(image),
                "dim": emb.get_embedding_dim(),
            }

        # --- Face detection + embedding ------------------------------------
        if _opts(spec.get("face")) is not None:
            faces = get_face().embed(image)
            debug_viewer.show_face_recognition(image, faces)
            data["face"] = [_serialize_face(f) for f in faces]

        # --- Appearance (attire) embed -------------------------------------
        if _opts(spec.get("appearance")) is not None:
            embedding = get_appearance().embed(image)
            debug_viewer.show_appearance(image, embedding)
            data["appearance"] = {"embedding": embedding}
    except Exception as exc:
        return error(str(exc), 500)

    return success(data)


@bp.post("/embed-text")
def embed_text():
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    if not text:
        return error("Missing 'text' field")
    try:
        emb = get_embedding()
        embedding = emb.embed_text(text)
        dim = emb.get_embedding_dim()
    except Exception as exc:
        return error(str(exc), 500)
    return success({"embedding": embedding, "dim": dim})
