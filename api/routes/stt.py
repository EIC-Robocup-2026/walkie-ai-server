"""STT (Speech-to-Text) blueprint — provider chosen in config.toml, loaded at startup."""

from flask import Blueprint, request

from api.routes.config import compact, section
from api.utils import error, success
from services.stt import STT

bp = Blueprint("stt", __name__, url_prefix="/stt")

# Provider + tuning come from [stt] in config.toml (default "whisper" — non-breaking).
# Set provider = "nemo" to use Chalk's local NemotronASR; [stt.<provider>] tunes the
# chosen provider (e.g. [stt.whisper] device/compute_type, [stt.nemo] model_path/device).
_cfg = section("stt")
_provider = _cfg.get("provider", "whisper")
# Pass the matching [stt.<provider>] sub-table straight through; compact() drops
# empty-string values so a provider's own defaults still apply when left unset.
_config = compact(_cfg.get(_provider, {}))

_stt = STT(provider=_provider, **_config)


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
