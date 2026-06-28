"""Flask application factory."""

from flask import Flask

from api.routes import grasp, image, stt, tts
from api.routes.config import section


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
                "object_detection": section("object_detection").get("provider", "yoloe"),
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

    # Pre-load GraspNet at startup so the first /grasp request runs at steady
    # state instead of paying the ~0.9 s lazy load + forward autotune on the
    # user's first call. Opt-in: GRASP_PRELOAD env wins, else [grasp].preload
    # (default off — keeps the lazy-load-on-first-call behaviour that saves VRAM
    # when manipulation isn't used).
    import os

    _env = os.getenv("GRASP_PRELOAD")
    if _env is not None:
        _preload = _env.strip().lower() in ("1", "true", "yes", "on")
    else:
        _preload = bool(section("grasp").get("preload", False))
    if _preload:
        from api.models import preload_grasp

        preload_grasp()

    return app
