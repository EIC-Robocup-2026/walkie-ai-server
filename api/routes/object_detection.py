"""Object Detection blueprint — YOLO provider, loaded at startup."""

from flask import Blueprint, request

from api.utils import error, image_from_request_file, mask_to_b64, pil_to_b64, success
from services.object_detection import ObjectDetection
from services.object_detection.base import DetectedObject

bp = Blueprint("object_detection", __name__, url_prefix="/object-detection")

_od = ObjectDetection(provider="yolo")
_od.load_model()


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
        detections = _od.detect(image)
    except Exception as exc:
        return error(str(exc), 500)

    return success([_serialize_detection(d) for d in detections])


def _serialize_detection(obj: DetectedObject) -> dict:
    return {
        "bbox": list(obj.bbox),
        "area_ratio": obj.area_ratio,
        "class_id": obj.class_id,
        "class_name": obj.class_name,
        "confidence": obj.confidence,
        "cropped_image_b64": pil_to_b64(obj.cropped_image),
        "mask_b64": mask_to_b64(obj.mask),
    }
