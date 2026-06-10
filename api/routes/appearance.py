"""Appearance (attire) re-ID blueprint — OSNet provider, lazy-loaded.

Stateless: one person crop in → one L2-normalized 512-d appearance vector
out. The agent crops to the person bbox before sending and owns enrollment,
fusion with the face embedding, thresholds, and the people database — the
server never stores or compares embeddings.

Pipeline by Chalk (EIC team); contract frozen in
``docs/appearance_service_handoff.md``.
"""

from flask import Blueprint, request

from api.utils import error, image_from_request_file, success
from services import debug_viewer
from services.appearance import Appearance

bp = Blueprint("appearance", __name__, url_prefix="/appearance")

_ap: Appearance | None = None


def _get_ap() -> Appearance:
    global _ap
    if _ap is None:
        _ap = Appearance(provider="osnet")
        _ap.load_model()
    return _ap


@bp.get("/providers")
def list_providers():
    return success(Appearance.available_providers())


@bp.get("/info")
def info():
    """Model provenance so the agent can detect a future model swap."""
    try:
        ap = _get_ap()
    except Exception as exc:  # e.g. torchreid not installed — keep the envelope
        return error(str(exc), 500)
    return success({"model_name": ap.get_model_name(), "dim": ap.get_embedding_dim()})


@bp.post("/embed")
def embed():
    if "image" not in request.files:
        return error("Missing 'image' file in request")

    try:
        image = image_from_request_file(request.files["image"])
    except Exception as exc:
        return error(f"Invalid image: {exc}")

    try:
        embedding = _get_ap().embed(image)
    except Exception as exc:
        return error(str(exc), 500)

    debug_viewer.show_appearance(image, embedding)
    return success({"embedding": embedding})
