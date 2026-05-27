"""Image Caption blueprint — PaliGemma provider, loaded at startup."""

from flask import Blueprint, request

from api.utils import error, image_from_request_file, success
from services import debug_viewer
from services.image_caption import ImageCaption

bp = Blueprint("image_caption", __name__, url_prefix="/image-caption")

_ic: ImageCaption | None = None


def _get_ic() -> ImageCaption:
    global _ic
    if _ic is None:
        _ic = ImageCaption(provider="florence2-base")
        _ic.load_model()
    return _ic


@bp.get("/providers")
def list_providers():
    return success(ImageCaption.available_providers())


@bp.post("/caption")
def caption():
    if "image" not in request.files:
        return error("Missing 'image' file in request")

    try:
        image = image_from_request_file(request.files["image"])
    except Exception as exc:
        return error(f"Invalid image: {exc}")

    prompt = request.form.get("prompt")

    try:
        result = _get_ic().caption(image, prompt=prompt)
    except Exception as exc:
        return error(str(exc), 500)

    debug_viewer.show_image_caption(image, result)
    return success({"caption": result})


@bp.post("/caption-batch")
def caption_batch():
    files = request.files.getlist("images")
    if not files:
        return error("Missing 'images' files in request")

    try:
        images = [image_from_request_file(f) for f in files]
    except Exception as exc:
        return error(f"Invalid image: {exc}")

    prompts_raw = request.form.getlist("prompts")
    prompts = prompts_raw if prompts_raw else None

    try:
        results = _get_ic().caption_batch(images, prompts=prompts)
    except Exception as exc:
        return error(str(exc), 500)

    if images and results:
        debug_viewer.show_image_caption(images[-1], results[-1])
    return success({"captions": results})
