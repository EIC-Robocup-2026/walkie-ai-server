"""STT (Speech-to-Text) blueprint — Whisper provider, loaded at startup."""

from flask import Blueprint, request

from api.utils import error, success
from services.stt import STT

bp = Blueprint("stt", __name__, url_prefix="/stt")

_stt = STT(provider="whisper")


@bp.get("/providers")
def list_providers():
    return success(STT.available_providers())


@bp.post("/transcribe")
def transcribe():
    if "audio" not in request.files:
        return error("Missing 'audio' file in request")

    audio_bytes = request.files["audio"].read()

    try:
        transcription = _stt.transcribe(audio_bytes)
    except Exception as exc:
        return error(str(exc), 500)

    return success({"transcription": transcription})
