"""InsightFace face-recognition provider (buffalo_l: RetinaFace + ArcFace).

``buffalo_l`` bundles a RetinaFace detector and an ArcFace (ResNet-100)
recognizer. ``FaceAnalysis.get()`` runs both in one call and hands back, per
face, an ``xyxy`` bbox, a detection score, and a 512-d **already L2-normalized**
``normed_embedding`` — exactly the contract the agent depends on, with no
fine-tuning or dataset required.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..base import FaceEmbedding, FaceRecognitionProvider

# ---------------------------------------------------------------------------
# Lazy import guard – avoid pulling in insightface/onnxruntime at module level
# (matches the YOLO providers, and keeps a missing/broken model from blocking
# server startup — the route lazy-loads on first request).
# ---------------------------------------------------------------------------
_insightface_imported = False


def _ensure_insightface() -> None:
    global _insightface_imported
    if _insightface_imported:
        return
    try:
        import insightface  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "InsightFace face provider requires insightface + onnxruntime. "
            "Install with: pip install insightface onnxruntime-gpu "
            "(or onnxruntime for CPU)."
        ) from e
    _insightface_imported = True


def _auto_ctx_id() -> int:
    """Return ``0`` (GPU) when onnxruntime exposes a CUDA provider, else ``-1``."""
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return 0
    except Exception:
        pass
    return -1


class InsightFaceProvider(FaceRecognitionProvider):
    """Face detection + recognition via InsightFace ``FaceAnalysis``."""

    DEFAULT_MODEL = "buffalo_l"
    DIM = 512

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the InsightFace provider.

        Args:
            config: Optional keys:
                - model: InsightFace model pack name (default ``"buffalo_l"``).
                - ctx_id: onnxruntime context id; ``0`` → GPU, ``-1`` → CPU.
                    Default: auto (GPU when a CUDA provider is available).
                - det_size: Detector input size as ``(w, h)`` (default ``(640, 640)``).
                - det_thresh: Detection threshold; faces below it are dropped by
                    the detector (default: InsightFace's own default).
        """
        self._config = config
        self._model_pack: str = config.get("model", self.DEFAULT_MODEL)
        ctx = config.get("ctx_id")
        self._ctx_id: int = int(ctx) if ctx is not None else _auto_ctx_id()
        det_size = config.get("det_size", (640, 640))
        self._det_size: tuple[int, int] = (int(det_size[0]), int(det_size[1]))
        self._det_thresh = config.get("det_thresh")
        self._app: Any | None = None
        self._model_name = f"insightface-{self._model_pack}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Pre-load the detector + recognizer into memory."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        _ensure_insightface()
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name=self._model_pack)
        prepare_kwargs: dict[str, Any] = {
            "ctx_id": self._ctx_id,
            "det_size": self._det_size,
        }
        if self._det_thresh is not None:
            prepare_kwargs["det_thresh"] = float(self._det_thresh)
        app.prepare(**prepare_kwargs)
        self._app = app

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def embed(self, image: Image.Image) -> list[FaceEmbedding]:
        """Detect faces and return one normalized embedding per face."""
        self._ensure_loaded()
        assert self._app is not None

        # InsightFace expects a BGR HxWx3 uint8 array (OpenCV convention).
        rgb = np.asarray(image.convert("RGB"))
        bgr = rgb[:, :, ::-1]

        faces = self._app.get(bgr)
        results: list[FaceEmbedding] = []
        for f in faces:
            results.append(
                FaceEmbedding(
                    bbox_xyxy=tuple(int(v) for v in f.bbox),
                    embedding=[float(x) for x in f.normed_embedding],
                    det_score=float(f.det_score),
                )
            )
        return results

    def get_model_name(self) -> str:
        return self._model_name

    def get_embedding_dim(self) -> int:
        return self.DIM
