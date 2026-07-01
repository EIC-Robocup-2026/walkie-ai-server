"""TTS (Text-to-Speech) blueprint — provider chosen in config.toml, lazy-loaded on first request."""

from flask import Blueprint, Response, request, stream_with_context

from api.routes.config import section
from api.utils import error, success
from services.tts import TTS

bp = Blueprint("tts", __name__, url_prefix="/tts")

_tts: TTS | None = None
_audio_content_type: str | None = None


def _infer_content_type(formats: list[str]) -> str:
    fmt_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "pcm": "audio/pcm",
    }
    for fmt in formats:
        mime = fmt_map.get(fmt.lower().split("_")[0])
        if mime:
            return mime
    return "application/octet-stream"


def _get_tts() -> tuple[TTS, str]:
    global _tts, _audio_content_type
    if _tts is None:
        # Provider + voice come from [tts] in config.toml; [tts.<provider>] holds
        # that provider's kwargs (piper: voice_path/voice_name, elevenlabs: voice_id).
        cfg = section("tts")
        provider = cfg.get("provider", "piper")
        params = dict(cfg.get(provider, {}))
        _tts = TTS(provider=provider, **params)
        _audio_content_type = _infer_content_type(_tts.get_supported_formats())
    assert _audio_content_type is not None
    return _tts, _audio_content_type


def preload() -> bool:
    """Warm the TTS provider (voice model) so the first /tts request runs hot.

    Called at startup by ``api.create_app`` when ``WALKIE_PRELOAD`` is on.
    Best-effort: a TTS load failure must not block boot — it prints a loud line
    and returns False so the caller's summary reflects it.
    """
    try:
        _get_tts()
        return True
    except Exception as exc:  # noqa: BLE001 — never block boot on TTS
        print(f"[preload] tts warm failed (non-fatal): {exc}")
        return False


@bp.get("/providers")
def list_providers():
    return success(TTS.available_providers())


@bp.post("/synthesize")
def synthesize():
    body = request.get_json(silent=True) or {}
    text = body.get("text") or request.form.get("text")
    if not text:
        return error("Missing 'text' field")

    try:
        tts, content_type = _get_tts()
        audio_bytes = tts.synthesize(text)
    except Exception as exc:
        return error(str(exc), 500)

    return Response(audio_bytes, mimetype=content_type)


@bp.post("/synthesize-stream")
def synthesize_stream():
    body = request.get_json(silent=True) or {}
    text = body.get("text") or request.form.get("text")
    if not text:
        return error("Missing 'text' field")

    try:
        tts, content_type = _get_tts()

        def generate():
            for chunk in tts.synthesize_stream(text):
                yield chunk
    except Exception as exc:
        return error(str(exc), 500)

    return Response(stream_with_context(generate()), mimetype=content_type)
