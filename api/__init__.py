"""Flask application factory."""

from flask import Flask

from api.routes import (
    appearance,
    face_recognition,
    image_caption,
    image_embed,
    object_detection,
    pose_estimation,
    stt,
    tts,
)


def create_app() -> Flask:
    app = Flask(__name__)

    app.register_blueprint(stt.bp)
    app.register_blueprint(tts.bp)
    app.register_blueprint(object_detection.bp)
    app.register_blueprint(pose_estimation.bp)
    app.register_blueprint(image_caption.bp)
    app.register_blueprint(image_embed.bp)
    app.register_blueprint(face_recognition.bp)
    app.register_blueprint(appearance.bp)

    @app.get("/")
    def index():
        return {
            "service": "walkie-agent-v2",
            "models": {
                "stt": "whisper",
                "tts": "elevenlabs",
                "object_detection": "yolo",
                "pose_estimation": "yolo_pose",
                "image_caption": "paligemma",
                "image_embed": "clip",
                "face_recognition": "insightface",
                "appearance": "osnet",
            },
            "endpoints": [
                "/stt/transcribe",
                "/tts/synthesize", "/tts/synthesize-stream",
                "/object-detection/detect",
                "/pose-estimation/estimate",
                "/image-caption/caption", "/image-caption/caption-batch",
                "/image-embed/embed-image", "/image-embed/embed-text", "/image-embed/similarity",
                "/face-recognition/embed", "/face-recognition/info",
                "/appearance/embed", "/appearance/info",
            ],
        }

    return app
