"""YOLO object detection provider (Objects365 pretrained)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

from ..base import DetectedObject, ObjectDetectionProvider

# Lazy imports to avoid loading torch/ultralytics until first use
_ultralytics_imported = False


def _ensure_ultralytics() -> None:
    global _ultralytics_imported
    if _ultralytics_imported:
        return
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "YOLO provider requires ultralytics. Install with: pip install ultralytics"
        ) from e
    _ultralytics_imported = True


def _get_model_path(config: dict[str, Any]) -> str:
    """Resolve model path: local file, or download from Hugging Face."""
    model = config.get("model", "yolo11n_object365")
    if os.path.isfile(model):
        return model
    # Try Hugging Face: NRtred/yolo11n_object365 has yolo11n_object365.pt
    if model in ("yolo11n_object365", "objects365") or "yolo11n" in model.lower():
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="NRtred/yolo11n_object365",
                filename="yolo11n_object365.pt",
            )
            return path
        except Exception as e:
            raise FileNotFoundError(
                "Could not download Objects365 YOLO model from Hugging Face. "
                "Install huggingface_hub and ensure HF_TOKEN is set if needed. "
                f"Error: {e}"
            ) from e
    # Assume it's a path or model name for YOLO() (e.g. "yolo11n.pt")
    return model


class YOLOObjectDetectionProvider(ObjectDetectionProvider):
    """Object detection via YOLO pretrained on Objects365 (365 classes)."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize YOLO provider.

        Args:
            config: Optional keys:
                - model: "yolo11n_object365" (default, downloads from HF), or path to .pt file
                - device: "cuda" or "cpu" (default: auto)
                - conf_threshold: Minimum confidence (0-1) to keep a detection (default: 0.25)
                - iou_threshold: NMS IOU threshold (default: 0.45)
                - max_objects: Maximum number of detections to return (default: 50)
                - crop_padding: Pixels to add around each crop (default: 10)
                - min_area_ratio: Filter out boxes smaller than this (default: 0.0005)
                - max_area_ratio: Filter out boxes larger than this (default: 0.95)
        """
        self._config = config

        device = config.get("device")
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._conf_threshold = float(config.get("conf_threshold", 0.25))
        self._iou_threshold = float(config.get("iou_threshold", 0.45))
        self._max_objects = int(config.get("max_objects", 50))
        self._crop_padding = int(config.get("crop_padding", 10))
        self._min_area_ratio = float(config.get("min_area_ratio", 0.0005))
        self._max_area_ratio = float(config.get("max_area_ratio", 0.95))
        self._model = None
        self._model_name = "yolo26m.pt"

    def load_model(self) -> None:
        """Pre-load YOLO model weights into memory."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Lazy-load YOLO model on first use."""
        if self._model is not None:
            return
        _ensure_ultralytics()
        from ultralytics import YOLO

        model_path = _get_model_path(self._config)
        self._model = YOLO(model_path)

    def detect(self, image: Image.Image) -> list[DetectedObject]:
        """Run YOLO inference and return detections as DetectedObject list."""
        self._ensure_loaded()
        assert self._model is not None
        img_rgb = np.array(image)
        if img_rgb.ndim == 2:
            img_rgb = np.stack([img_rgb] * 3, axis=-1)
        h, w = img_rgb.shape[0], img_rgb.shape[1]
        total_area = h * w

        # Ultralytics expects BGR or RGB; PIL is RGB. YOLO accepts numpy HWC.
        results = self._model.predict(
            img_rgb,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            verbose=False,
            device=self._device,
        )

        detections: list[DetectedObject] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        # boxes.xyxy is (N, 4) in pixel coords (x1, y1, x2, y2)
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        names = getattr(self._model, "names", {}) or {}
        if isinstance(names, list):
            names = {i: names[i] for i in range(len(names))}

        # Sort by area descending to prioritize larger objects
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        order = np.argsort(-areas)
        pad = self._crop_padding

        for idx in order:
            if len(detections) >= self._max_objects:
                break
            x1, y1, x2, y2 = xyxy[idx]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            area_ratio = (x2 - x1) * (y2 - y1) / total_area
            if area_ratio < self._min_area_ratio or area_ratio > self._max_area_ratio:
                continue
            # Clamp and add padding for crop
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)
            if x2p <= x1p or y2p <= y1p:
                continue
            crop_rgb = img_rgb[y1p:y2p, x1p:x2p]
            crop_pil = Image.fromarray(crop_rgb)
            # bbox = (x1p, y1p, x2p, y2p)
            bbox = ((x1p+x2p)//2, (y1p+y2p)//2, abs(x2p - x1p), abs(y2p - y1p))  # cx, cy, w, h
            class_id = int(cls_ids[idx])
            class_name = names.get(class_id, "unknown")
            confidence = float(confs[idx])
            detections.append(
                DetectedObject(
                    mask=None,
                    bbox=[x1p, y1p, x2p, y2p],
                    area_ratio=area_ratio,
                    cropped_image=crop_pil,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                )
            )
        return detections

    def get_model_name(self) -> str:
        return self._model_name
