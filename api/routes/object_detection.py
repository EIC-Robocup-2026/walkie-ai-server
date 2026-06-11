"""Object Detection blueprint.

Provider is selectable via the ``OBJECT_DETECTION_PROVIDER`` env var:
  - "yolo" (default): fixed 365-class Objects365 detector, no masks.
  - "sam3": open-vocabulary concept segmentation (text prompts + masks; with
    no prompts it segments everything as unlabeled instances).
  - "yoloe": open-vocabulary detection + segmentation (text prompts or a
    prompt-free model when no prompts are given).

For SAM3 the checkpoint path can be set via ``SAM3_MODEL`` (weights must be
downloaded manually). For YOLOE the text / prompt-free checkpoints can be set
via ``YOLOE_MODEL`` / ``YOLOE_PF_MODEL`` (both auto-download by default).

Requests may pass optional ``prompts`` (comma-separated or repeated form field)
to steer SAM3 / YOLOE; YOLO ignores them. Pass ``return_mask=true`` to include a
base64 segmentation ``mask_b64`` per detection (where the provider supports it).
"""

import os

from flask import Blueprint, request

from api.utils import error, image_from_request_file, mask_to_b64, pil_to_b64, success
from services import debug_viewer
from services.object_detection import ObjectDetection
from services.object_detection.base import DetectedObject

bp = Blueprint("object_detection", __name__, url_prefix="/object-detection")

_PROVIDER = os.environ.get("OBJECT_DETECTION_PROVIDER", "yoloe")
_provider_config: dict = {}
if _PROVIDER == "yolo":
    _provider_config["model"] = os.environ.get("YOLO_MODEL", "YOLO26l-seg")  # Optional custom model path  
if _PROVIDER == "sam3" and os.environ.get("SAM3_MODEL"):
    _provider_config["model"] = os.environ["SAM3_MODEL"]
if _PROVIDER == "yoloe":
    if os.environ.get("YOLOE_MODEL"):
        _provider_config["model"] = os.environ["YOLOE_MODEL"]
    if os.environ.get("YOLOE_PF_MODEL"):
        _provider_config["pf_model"] = os.environ["YOLOE_PF_MODEL"]

_od = ObjectDetection(provider=_PROVIDER, **_provider_config)
_od.load_model()


def _prompts_from_request() -> list[str] | None:
    """Read optional text prompts from the request (SAM3 / YOLOE only).

    Accepts repeated ``prompts`` fields or a single comma-separated value.
    """
    values = request.form.getlist("prompts")
    if len(values) == 1 and "," in values[0]:
        values = values[0].split(",")
    cleaned = [v.strip() for v in values if v and v.strip()]
    return cleaned or None


def _return_mask_from_request() -> bool:
    """Read the optional ``return_mask`` flag from the request (default False)."""
    val = (request.form.get("return_mask") or "").strip().lower()
    return val in ("1", "true", "yes", "on")


@bp.get("/providers")
def list_providers():
    return success(ObjectDetection.available_providers())


@bp.post("/detect")
def detect():
    if "image" not in request.files:
        return error("Missing 'image' file in request")

    try:
        image = image_from_request_file(request.files["image"])
    except Exception as exc:
        return error(f"Invalid image: {exc}")

    try:
        detections = _od.detect(
            image,
            _prompts_from_request(),
            return_mask=_return_mask_from_request(),
        )
    except Exception as exc:
        return error(str(exc), 500)

    debug_viewer.show_object_detection(image, detections)
    return success([_serialize_detection(d) for d in detections])


def _serialize_detection(obj: DetectedObject) -> dict:
    return {
        "bbox": list(obj.bbox),
        "area_ratio": obj.area_ratio,
        "class_id": obj.class_id,
        "class_name": obj.class_name,
        "confidence": obj.confidence,
        "mask_b64": mask_to_b64(obj.mask),
    }
