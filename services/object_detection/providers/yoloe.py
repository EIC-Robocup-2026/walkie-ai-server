"""YOLOE object detection provider (open-vocabulary detect + segment).

YOLOE (Ultralytics) performs open-vocabulary detection and instance
segmentation. Two checkpoints are used:
  - a text-prompt segmentation model (default ``yoloe-11m-seg.pt``): used when
    the request supplies text prompts; the prompt list is applied via
    ``set_classes`` so the model detects exactly those concepts.
  - a prompt-free segmentation model (default ``yoloe-11m-seg-pf.pt``): used
    when NO prompts are supplied — it detects from a built-in open vocabulary.

Unlike SAM3, the YOLOE checkpoints auto-download from the Ultralytics asset
registry on first use; no manual / Hugging Face download is needed.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

from ..base import (
    DetectedObject,
    ObjectDetectionProvider,
    bbox_for,
    label_for,
    resize_mask,
)

# Lazy import to avoid loading torch/ultralytics until first use.
_ultralytics_imported = False


def _ensure_ultralytics() -> None:
    global _ultralytics_imported
    if _ultralytics_imported:
        return
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "YOLOE provider requires ultralytics. Install/upgrade with: "
            "pip install -U ultralytics"
        ) from e
    _ultralytics_imported = True


class YOLOEObjectDetectionProvider(ObjectDetectionProvider):
    """Open-vocabulary detection + segmentation via Ultralytics YOLOE."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize YOLOE provider.

        Args:
            config: Optional keys:
                - model: Text-prompt seg checkpoint (default "yoloe-11m-seg.pt").
                  Auto-downloads. Used when a request supplies prompts.
                - pf_model: Prompt-free seg checkpoint (default
                  "yoloe-11m-seg-pf.pt"). Auto-downloads. Used when a request
                  supplies no prompts (open-vocabulary).
                - device: "cuda" or "cpu" (default: auto).
                - half: FP16 inference (default: False; see __init__ note --
                  this ultralytics version's seg mask postprocess is fp32-only).
                - prompts: Default text concepts (list[str]) applied to the text
                  model when a request omits prompts. If unset, requests with no
                  prompts use the prompt-free model instead.
                - conf_threshold: Minimum confidence (0-1) to keep (default: 0.25).
                - iou_threshold: NMS IOU threshold (default: 0.45).
                - max_objects: Maximum detections to return (default: 50).
                - crop_padding: Pixels added around each crop bbox (default: 10).
                - min_area_ratio: Drop boxes smaller than this (default: 0.0005).
                - max_area_ratio: Drop boxes larger than this (default: 0.95).
                - imgsz: Inference image size (default: 640). None uses the model
                  default.
                - preload: Eagerly load both checkpoints in ``load_model``
                  (default: True). Set False to lazy-load each on first use.
        """
        self._config = config

        device = config.get("device")
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"YOLOE provider using device: {device or 'auto'}")
        self._device = device
        # FP16 is OFF by default. Every YOLOE checkpoint we serve is a
        # segmentation (-seg) model, so Ultralytics always runs its mask
        # postprocess (ops.process_mask) -- even when return_mask is False --
        # and this version matmuls the fp16 mask coefficients against an fp32
        # `protos.float()`, raising "mat1 and mat2 ... Half != float" on every
        # predict. Re-enable with `half = true` in config only on an
        # ultralytics build that fixes process_mask. Half buys little here (the
        # model is small) and costs only a touch of speed/memory in fp32.
        self._half = bool(config.get("half", False))

        prompts = config.get("prompts")
        self._default_prompts: list[str] = list(prompts) if prompts else []

        self._conf_threshold = float(config.get("conf_threshold", 0.25))
        self._iou_threshold = float(config.get("iou_threshold", 0.45))
        self._max_objects = int(config.get("max_objects", 50))
        self._crop_padding = int(config.get("crop_padding", 10))
        self._min_area_ratio = float(config.get("min_area_ratio", 0.0005))
        self._max_area_ratio = float(config.get("max_area_ratio", 0.95))
        self._imgsz = config.get("imgsz", 640)
        self._preload = bool(config.get("preload", True))

        self._model = None  # text-prompt model
        self._pf_model = None  # prompt-free model
        self._current_classes: list[str] | None = None
        self._model_name = os.path.basename(
            str(config.get("model", "yoloe-11m-seg.pt"))
        )

    def load_model(self) -> None:
        """Pre-load YOLOE checkpoint(s) into memory."""
        if self._preload:
            self._ensure_text_loaded()
            self._ensure_pf_loaded()

    def _ensure_text_loaded(self) -> None:
        """Lazy-load the text-prompt YOLOE model on first use."""
        if self._model is not None:
            return
        _ensure_ultralytics()
        from ultralytics import YOLOE

        self._model = YOLOE(self._config.get("model", "yoloe-11m-seg.pt"))

    def _ensure_pf_loaded(self) -> None:
        """Lazy-load the prompt-free YOLOE model on first use."""
        if self._pf_model is not None:
            return
        _ensure_ultralytics()
        from ultralytics import YOLOE

        self._pf_model = YOLOE(
            self._config.get("pf_model", "yoloe-11m-seg-pf.pt")
        )

    def _apply_classes(self, classes: list[str]) -> None:
        """Set the text-prompt vocabulary (no-op when unchanged).

        Setting classes encodes the prompt text once; an app-level cache avoids
        re-encoding when the same prompt set is reused across requests.

        The text encoder always emits float32 embeddings, but a prior
        ``predict(half=True)`` converts the model head -- including the reprta
        block that refines those embeddings -- to fp16 in place. Re-encoding
        then multiplies float32 features by fp16 weights, raising "mat1 and mat2
        must have the same dtype, but got Float and Half". Float the model for
        the encode and restore half afterwards; predict re-casts the stored
        embeddings to the input dtype, so detection precision is unchanged.
        """
        if self._current_classes == classes:
            return
        assert self._model is not None

        import torch

        torch_model = getattr(self._model, "model", None)
        was_half = torch_model is not None and any(
            p.dtype == torch.float16 for p in torch_model.parameters()
        )
        if was_half:
            torch_model.float()
        try:
            try:
                self._model.set_classes(classes)
            except Exception:
                # Robustness across ultralytics versions: pass text embeddings too.
                self._model.set_classes(classes, self._model.get_text_pe(classes))
        finally:
            if was_half:
                torch_model.half()
        self._current_classes = list(classes)

    def detect(
        self,
        image: Image.Image,
        prompts: list[str] | None = None,
        return_mask: bool = False,
    ) -> list[DetectedObject]:
        """Run YOLOE inference and return a DetectedObject list.

        When ``prompts`` (or the configured default prompts) are present the
        text-prompt model is used with ``set_classes``. Otherwise the
        prompt-free model is used (built-in open vocabulary).

        Args:
            image: PIL Image (RGB).
            prompts: Open-vocabulary noun phrases to detect.
            return_mask: When True, include the segmentation mask on each
                detection; otherwise only bounding boxes are returned.
        """
        concepts = [p for p in (prompts or self._default_prompts) if p and p.strip()]

        img_rgb = np.array(image)
        if img_rgb.ndim == 2:
            img_rgb = np.stack([img_rgb] * 3, axis=-1)
        h, w = img_rgb.shape[0], img_rgb.shape[1]
        total_area = float(h * w)

        predict_kwargs: dict = dict(
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            verbose=False,
            device=self._device,
        )
        if self._imgsz is not None:
            predict_kwargs["imgsz"] = int(self._imgsz)
        if self._half:
            predict_kwargs["half"] = True
        if return_mask:
            predict_kwargs["retina_masks"] = True

        if concepts:
            self._ensure_text_loaded()
            self._apply_classes(concepts)
            model = self._model
        else:
            # Prompt-free model detects from its built-in vocabulary; it must
            # NOT receive set_classes.
            self._ensure_pf_loaded()
            model = self._pf_model
        assert model is not None

        results = model.predict(img_rgb, **predict_kwargs)
        return self._parse_results(results, concepts, w, h, total_area, return_mask)

    def _parse_results(
        self,
        results: Any,
        concepts: list[str],
        w: int,
        h: int,
        total_area: float,
        return_mask: bool,
    ) -> list[DetectedObject]:
        """Convert Ultralytics YOLOE results into a DetectedObject list."""
        if not results:
            return []
        r = results[0] if isinstance(results, (list, tuple)) else results

        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        # .float() guards against float16 overflow on half-precision models.
        xyxy = boxes.xyxy.float().cpu().numpy()
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else None
        names = getattr(r, "names", None)

        mask_data = None
        if return_mask:
            masks = getattr(r, "masks", None)
            if masks is not None and getattr(masks, "data", None) is not None:
                mask_data = masks.data.float().cpu().numpy()  # (N, mh, mw)

        # Order by area descending (prefer larger objects), like other providers.
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        order = np.argsort(-areas)
        pad = self._crop_padding

        detections: list[DetectedObject] = []
        for idx in order:
            if len(detections) >= self._max_objects:
                break

            mask_2d = None
            if mask_data is not None and idx < mask_data.shape[0]:
                mask_2d = resize_mask(mask_data[idx], w, h)

            bbox = bbox_for(xyxy, idx, mask_2d, w, h)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            area_ratio = (x2 - x1) * (y2 - y1) / total_area
            if area_ratio < self._min_area_ratio or area_ratio > self._max_area_ratio:
                continue

            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)
            if x2p <= x1p or y2p <= y1p:
                continue

            confidence = float(conf[idx]) if conf is not None else None
            class_name = label_for(cls, idx, names, concepts)
            class_id = int(cls[idx]) if cls is not None else None

            detections.append(
                DetectedObject(
                    mask=mask_2d if return_mask else None,
                    bbox=[x1p, y1p, x2p, y2p],
                    area_ratio=area_ratio,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                )
            )
        return detections

    def get_model_name(self) -> str:
        return self._model_name
