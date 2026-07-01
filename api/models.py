"""Shared model singletons for the image-processing endpoints.

Owns every heavy vision model behind one accessor each so the unified
``/image`` blueprint reuses the same loaded instance instead of each route
holding its own. Detection / pose / image-embed load **eagerly** at import
(always on the critical path); caption / face / appearance load **lazily** on
first use (heavier / optional, so a broken one never blocks import). When
``WALKIE_PRELOAD`` is on (the default) ``api.create_app`` warms those lazy
singletons at startup via :func:`preload_lazy_singletons`, so the first request
to every endpoint runs hot instead of paying the model load. Provider selection
mirrors the per-task ``config.toml`` tables that the old per-route modules read.
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


def preload_lazy_singletons() -> list[tuple[str, bool]]:
    """Eagerly warm the lazy vision singletons (caption / face / appearance).

    Without this each loads on the first request that touches it, so that first
    call pays the one-time model load. Warming them at boot moves that cost to
    startup and keeps first requests hot.

    Best-effort **per model**: these are the "heavier / optional" singletons, so
    one failing to load (e.g. a missing pack) must not take down the endpoints
    that already loaded eagerly at import. Each failure prints a loud line;
    ``api.create_app`` prints a one-line summary of what warmed vs failed.

    Returns:
        ``(name, ok)`` for each singleton, in warm order.
    """
    results: list[tuple[str, bool]] = []
    for name, getter in (
        ("caption", get_caption),
        ("face", get_face),
        ("appearance", get_appearance),
    ):
        try:
            getter()
            results.append((name, True))
        except Exception as exc:  # noqa: BLE001 — one optional model must not block boot
            print(f"[preload] {name} warm failed (non-fatal): {exc}")
            results.append((name, False))
    return results


def preload_grasp(warmup_runs: int = 3) -> None:
    """Eagerly load GraspNet and warm the forward at startup.

    Without this, GraspNet lazy-loads on the first ``/grasp`` call, so that first
    request pays the one-time model load + first-forward autotune (~0.9 s on top
    of the normal ~40 ms). Pre-loading moves that cost to boot. ``load_model()``
    already runs one warmup forward; the extra ``warmup_runs`` settle
    cuDNN/cuBLAS autotune for the real ``(num_point, 3)`` shape. Best-effort —
    warmup failures never block startup. Enabled via ``[grasp].preload`` /
    ``GRASP_PRELOAD`` (see ``api.create_app``).
    """
    import numpy as np

    g = get_grasp()
    prov = g.provider
    npoint = int(getattr(prov, "_num_point", 10000))
    dummy = np.random.randn(npoint, 3).astype(np.float32)
    for _ in range(max(0, warmup_runs)):
        try:
            prov._run_graspnet(dummy)
        except Exception as exc:  # noqa: BLE001 — never block startup on warmup
            print(f"[models] grasp warmup forward failed (non-fatal): {exc}")
            break
