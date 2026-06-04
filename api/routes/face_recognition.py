"""Face Recognition blueprint — InsightFace provider, lazy-loaded.

Stateless: image in → faces (xyxy bbox + L2-normalized embedding + det score)
out. Enrollment, names, and matching all live in the agent repo.
"""

from flask import Blueprint, request

from api.utils import error, image_from_request_file, success
from services import debug_viewer
from services.face_recognition import FaceRecognition
from services.face_recognition.base import FaceEmbedding

bp = Blueprint("face_recognition", __name__, url_prefix="/face-recognition")

_fr: FaceRecognition | None = None


def _get_fr() -> FaceRecognition:
    global _fr
    if _fr is None:
        _fr = FaceRecognition(provider="insightface")
        _fr.load_model()
    return _fr


@bp.get("/providers")
def list_providers():
    return success(FaceRecognition.available_providers())


@bp.get("/info")
def info():
    """Model provenance so the agent can detect a future model swap."""
    fr = _get_fr()
    return success({"model_name": fr.get_model_name(), "dim": fr.get_embedding_dim()})


@bp.post("/embed")
def embed():
    if "image" not in request.files:
        return error("Missing 'image' file in request")

    try:
        image = image_from_request_file(request.files["image"])
    except Exception as exc:
        return error(f"Invalid image: {exc}")

    try:
        faces = _get_fr().embed(image)
    except Exception as exc:
        return error(str(exc), 500)

    debug_viewer.show_face_recognition(image, faces)
    return success([_serialize_face(f) for f in faces])


def _serialize_face(face: FaceEmbedding) -> dict:
    return {
        "bbox_xyxy": list(face.bbox_xyxy),
        "embedding": face.embedding,
        "det_score": face.det_score,
    }
