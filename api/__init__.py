"""Flask application factory."""

import os

from flask import Flask

from api.routes import grasp, image, stt, tts
from api.routes.config import section


def _truthy(value: str | None) -> bool:
    """Parse an on/off env flag (unset -> False)."""
    return value is not None and value.strip().lower() in ("1", "true", "yes", "on")


def create_app() -> Flask:
    app = Flask(__name__)

    app.register_blueprint(stt.bp)
    app.register_blueprint(tts.bp)
    app.register_blueprint(image.bp)
    app.register_blueprint(grasp.bp)

    @app.get("/")
    def index():
        return {
            "service": "walkie-agent-v2",
            "models": {
                "stt": section("stt").get("provider", "whisper"),
                "tts": section("tts").get("provider", "piper"),
                "object_detection": section("object_detection").get("provider", "detecty"),
                "pose_estimation": section("pose_estimation").get("provider", "yolo_pose"),
                "image_caption": section("image_caption").get("provider", "florence2-base"),
                "image_embed": section("image_embed").get("provider", "clip"),
                "face_recognition": section("face_recognition").get("provider", "insightface"),
                "appearance": section("appearance").get("provider", "osnet"),
                "grasp": section("grasp").get("provider", "graspnet"),
            },
            "endpoints": [
                "/stt/transcribe",
                "/tts/synthesize", "/tts/synthesize-stream",
                "/image/process", "/image/embed-text",
                "/grasp",
            ],
        }

    # ------------------------------------------------------------------
    # Preload every model at startup so the first request to each endpoint runs
    # hot instead of paying its model load. Detection / pose / image-embed
    # (api.models import) and STT (stt route import) already loaded eagerly
    # above; this warms the remaining lazy singletons — caption, face,
    # appearance, TTS — plus GraspNet.
    #
    # WALKIE_PRELOAD is the master gate (default on). Set it to 0 to keep the
    # heavy / optional models lazy on a low-VRAM dev box; being the master gate
    # it also suppresses the grasp preload below regardless of grasp's own flag.
    # ------------------------------------------------------------------
    if _truthy(os.getenv("WALKIE_PRELOAD", "1")):
        from api import models

        results = models.preload_lazy_singletons()
        results.append(("tts", tts.preload()))

        # GraspNet is heavier (its own VRAM + pointnet2/knn CUDA ops) and warms
        # its forward, so it keeps a dedicated sub-gate: GRASP_PRELOAD env wins,
        # else [grasp].preload. Only consulted while the master gate is on.
        _genv = os.getenv("GRASP_PRELOAD")
        _grasp_preload = _truthy(_genv) if _genv is not None else bool(section("grasp").get("preload", False))
        if _grasp_preload:
            try:
                models.preload_grasp()
                results.append(("grasp", True))
            except Exception as exc:  # noqa: BLE001 — never block boot on grasp
                print(f"[preload] grasp warm failed (non-fatal): {exc}")
                results.append(("grasp", False))

        warmed = [name for name, ok in results if ok]
        failed = [name for name, ok in results if not ok]
        print(
            f"[preload] warmed: {', '.join(warmed) or 'none'}"
            f" — failed: {', '.join(failed) or 'none'}"
        )

    return app
