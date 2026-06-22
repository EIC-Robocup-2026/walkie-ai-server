"""Shared model singletons for the image-processing endpoints.

Owns every heavy vision model behind one accessor each so the unified
``/image`` blueprint reuses the same loaded instance instead of each route
holding its own. Detection / pose / image-embed load **eagerly** at import
(always on the critical path); caption / face / appearance load **lazily** on
first use (heavier or optional). Provider selection mirrors the per-task
``config.toml`` tables that the old per-route modules read.
"""

from __future__ import annotations

from api.routes.config import compact, section
from services.appearance import Appearance
from services.face_recognition import FaceRecognition
from services.grasp import Grasp
from services.image_caption import ImageCaption
from services.image_embed import Embedding
from services.object_detection import ObjectDetection
from services.pose_estimation import PoseEstimation

# --- Eager: detection / pose / image-embed (loaded at import) --------------

_od_cfg = section("object_detection")
_od_provider = _od_cfg.get("provider", "yoloe")
# The [object_detection.<provider>] sub-table holds that provider's kwargs;
# empty strings are dropped so the provider default is used.
_object_detection = ObjectDetection(
    provider=_od_provider, **compact(_od_cfg.get(_od_provider, {}))
)
_object_detection.load_model()

_pose = PoseEstimation(provider=section("pose_estimation").get("provider", "yolo_pose"))
_pose.load_model()

_embedding = Embedding(provider=section("image_embed").get("provider", "clip"))
_embedding.load_model()


def get_object_detection() -> ObjectDetection:
    return _object_detection


def get_pose() -> PoseEstimation:
    return _pose


def get_embedding() -> Embedding:
    return _embedding


# --- Lazy: caption / face / appearance (loaded on first use) ---------------

_caption: ImageCaption | None = None
_face: FaceRecognition | None = None
_appearance: Appearance | None = None
_grasp: Grasp | None = None


def get_caption() -> ImageCaption:
    global _caption
    if _caption is None:
        _caption = ImageCaption(
            provider=section("image_caption").get("provider", "florence2-base")
        )
        _caption.load_model()
    return _caption


def get_face() -> FaceRecognition:
    global _face
    if _face is None:
        _face = FaceRecognition(
            provider=section("face_recognition").get("provider", "insightface")
        )
        _face.load_model()
    return _face


def get_appearance() -> Appearance:
    global _appearance
    if _appearance is None:
        _appearance = Appearance(
            provider=section("appearance").get("provider", "osnet")
        )
        _appearance.load_model()
    return _appearance


def get_grasp() -> Grasp:
    global _grasp
    if _grasp is None:
        # GraspNet is heavy (its own VRAM + pointnet2/knn CUDA ops) and only needed
        # for manipulation, so it loads on first /grasp call. The [grasp.<provider>]
        # sub-table holds the provider's kwargs (checkpoint/root/device/tunables).
        cfg = section("grasp")
        provider = cfg.get("provider", "graspnet")
        _grasp = Grasp(provider=provider, **compact(cfg.get(provider, {})))
        _grasp.load_model()
    return _grasp
