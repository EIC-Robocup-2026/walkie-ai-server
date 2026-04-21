"""Pose Estimation blueprint — YOLO-Pose provider, loaded at startup."""

from flask import Blueprint, request

from api.utils import error, image_from_request_file, pil_to_b64, success
from services.pose_estimation import PoseEstimation
from services.pose_estimation.base import PersonPose, PoseKeypoint

bp = Blueprint("pose_estimation", __name__, url_prefix="/pose-estimation")

_pe = PoseEstimation(provider="yolo_pose")
_pe.load_model()


@bp.get("/providers")
def list_providers():
    return success(PoseEstimation.available_providers())


@bp.post("/estimate")
def estimate():
    if "image" not in request.files:
        return error("Missing 'image' file in request")

    try:
        image = image_from_request_file(request.files["image"])
    except Exception as exc:
        return error(f"Invalid image: {exc}")

    try:
        poses = _pe.estimate(image)
    except Exception as exc:
        return error(str(exc), 500)

    return success([_serialize_pose(p) for p in poses])


def _serialize_keypoint(kp: PoseKeypoint) -> dict:
    return {
        "index": kp.index,
        "name": kp.name,
        "x": kp.x,
        "y": kp.y,
        "confidence": kp.confidence,
    }


def _serialize_pose(pose: PersonPose) -> dict:
    return {
        "bbox": list(pose.bbox),
        "confidence": pose.confidence,
        "keypoints": [_serialize_keypoint(kp) for kp in pose.keypoints],
        "cropped_image_b64": pil_to_b64(pose.cropped_image),
    }
