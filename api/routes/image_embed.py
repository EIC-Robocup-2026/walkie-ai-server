"""Image Embedding blueprint — CLIP provider, loaded at startup."""

from flask import Blueprint, request

from api.utils import error, image_from_request_file, success
from services.image_embed import Embedding

bp = Blueprint("image_embed", __name__, url_prefix="/image-embed")

_emb = Embedding(provider="clip")
_emb.load_model()


@bp.get("/providers")
def list_providers():
    return success(Embedding.available_providers())


@bp.post("/embed-image")
def embed_image():
    if "image" not in request.files:
        return error("Missing 'image' file in request")

    try:
        image = image_from_request_file(request.files["image"])
    except Exception as exc:
        return error(f"Invalid image: {exc}")

    try:
        embedding = _emb.embed_image(image)
        dim = _emb.get_embedding_dim()
    except Exception as exc:
        return error(str(exc), 500)

    return success({"embedding": embedding, "dim": dim})


@bp.post("/embed-text")
def embed_text():
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    if not text:
        return error("Missing 'text' field")

    try:
        embedding = _emb.embed_text(text)
        dim = _emb.get_embedding_dim()
    except Exception as exc:
        return error(str(exc), 500)

    return success({"embedding": embedding, "dim": dim})


@bp.post("/similarity")
def similarity():
    """Compute cosine similarity between an image and a text query."""
    if "image" not in request.files:
        return error("Missing 'image' file in request")

    text = request.form.get("text")
    if not text:
        return error("Missing 'text' form field")

    try:
        image = image_from_request_file(request.files["image"])
    except Exception as exc:
        return error(f"Invalid image: {exc}")

    try:
        img_vec = _emb.embed_image(image)
        txt_vec = _emb.embed_text(text)
        score = _emb.similarity(img_vec, txt_vec)
    except Exception as exc:
        return error(str(exc), 500)

    return success({"similarity": score})
