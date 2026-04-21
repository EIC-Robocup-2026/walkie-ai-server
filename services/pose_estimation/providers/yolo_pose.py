"""YOLO pose estimation provider (YOLO26m-pose)."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..base import COCO_KEYPOINT_NAMES, PersonPose, PoseEstimationProvider, PoseKeypoint

# ---------------------------------------------------------------------------
# Lazy import guard – avoid pulling in torch/ultralytics at module level.
# ---------------------------------------------------------------------------
_ultralytics_imported = False


def _ensure_ultralytics() -> None:
    global _ultralytics_imported
    if _ultralytics_imported:
        return
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "YOLO pose provider requires ultralytics. "
            "Install with: pip install ultralytics"
        ) from e
    _ultralytics_imported = True


class YOLOPoseEstimationProvider(PoseEstimationProvider):
    """Pose estimation via YOLO26m-pose (ultralytics).

    The model detects persons **and** their 17 COCO keypoints in a single
    forward pass, so no separate person-detection step is needed.
    """

    DEFAULT_MODEL = "yolo26m-pose.pt"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the YOLO pose provider.

        Args:
            config: Optional keys:
                - model: Model name or path (default ``"yolo26m-pose.pt"``).
                    Ultralytics will auto-download if not present locally.
                - device: ``"cuda"`` or ``"cpu"`` (default: auto).
                - conf_threshold: Min person confidence (default 0.25).
                - iou_threshold: NMS IoU threshold (default 0.45).
                - max_persons: Max persons to return (default 20).
                - crop_padding: Pixels of padding around each person crop
                    (default 10).
                - keypoint_confidence_threshold: Min confidence for a keypoint
                    to be considered visible (default 0.5).
        """
        self._config = config

        device = config.get("device")
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device: str = device

        self._model_path: str = config.get("model", self.DEFAULT_MODEL)
        self._conf_threshold: float = float(config.get("conf_threshold", 0.25))
        self._iou_threshold: float = float(config.get("iou_threshold", 0.45))
        self._max_persons: int = int(config.get("max_persons", 20))
        self._crop_padding: int = int(config.get("crop_padding", 10))
        self._kpt_conf_threshold: float = float(
            config.get("keypoint_confidence_threshold", 0.5)
        )
        self._model: Any | None = None
        self._model_name = "yolo26s-pose"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Pre-load YOLO pose model weights into memory."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _ensure_ultralytics()
        from ultralytics import YOLO

        self._model = YOLO(self._model_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def estimate(self, image: Image.Image) -> list[PersonPose]:
        """Run YOLO pose inference and return detected persons with keypoints."""
        self._ensure_loaded()
        assert self._model is not None

        img_rgb = np.array(image)
        if img_rgb.ndim == 2:
            img_rgb = np.stack([img_rgb] * 3, axis=-1)
        h, w = img_rgb.shape[:2]

        results = self._model.predict(
            img_rgb,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            verbose=False,
            device=self._device,
        )

        persons: list[PersonPose] = []
        if not results:
            return persons

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return persons

        # boxes.xyxy  -> (N, 4) in pixel coords (x1, y1, x2, y2)
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        # keypoints.data -> (N, 17, 3)  where last dim is (x, y, conf)
        kpts_data: np.ndarray | None = None
        if r.keypoints is not None and r.keypoints.data is not None:
            kpts_data = r.keypoints.data.cpu().numpy()

        # Sort by area descending (bigger persons first).
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        order = np.argsort(-areas)
        pad = self._crop_padding

        for idx in order:
            if len(persons) >= self._max_persons:
                break

            x1, y1, x2, y2 = xyxy[idx]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = float(confs[idx])

            # Convert to (cx, cy, w, h) to match existing YOLO bbox convention.
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)
            if x2p <= x1p or y2p <= y1p:
                continue

            bbox = (
                (x1p + x2p) // 2,
                (y1p + y2p) // 2,
                abs(x2p - x1p),
                abs(y2p - y1p),
            )

            # Crop the person region.
            crop_rgb = img_rgb[y1p:y2p, x1p:x2p]
            crop_pil = Image.fromarray(crop_rgb)

            # Build keypoints list.
            keypoints: list[PoseKeypoint] = []
            if kpts_data is not None:
                kpt_arr = kpts_data[idx]  # (17, 3)
                for ki in range(kpt_arr.shape[0]):
                    kx, ky, kc = float(kpt_arr[ki, 0]), float(kpt_arr[ki, 1]), float(kpt_arr[ki, 2])
                    name = COCO_KEYPOINT_NAMES[ki] if ki < len(COCO_KEYPOINT_NAMES) else f"kpt_{ki}"
                    keypoints.append(
                        PoseKeypoint(x=kx, y=ky, confidence=kc, name=name, index=ki)
                    )

            persons.append(
                PersonPose(
                    bbox=bbox,
                    confidence=confidence,
                    keypoints=keypoints,
                    cropped_image=crop_pil,
                )
            )

        return persons

    def get_model_name(self) -> str:
        return self._model_name
