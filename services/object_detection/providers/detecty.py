"""detecty object detection provider (custom RoboCup@Home ensemble).

Wraps the vendored ``detecty`` package (``third_party/detecty``): a detector
tuned for the 30 official Incheon2026 objects. It decouples localization from
classification:

  * localize  — Grounding DINO finds class-agnostic boxes.
  * classify  — each crop is scored by an ensemble (DINOv3-L nearest-prototype
    + masked-HSV colour + EasyOCR brand text), which resolves the brand/colour
    "twins" (coke vs red_bull, pepsi vs soju, red vs yellow bell pepper) that
    generic detectors get wrong.

detecty ships no prototype bank; ``prototypes.npz`` is built once from the
bundled reference images (``prototypes/`` + ``objects_gt/``) using the DINOv3
embedder and then cached. detecty produces bounding boxes only — no masks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..base import DetectedObject, ObjectDetectionProvider

# Repo root: services/object_detection/providers/detecty.py -> up 3 = repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Default locations (repo-relative), shared with scripts/prefetch_detecty.py so
# the warm-up script and the provider build/load the same prototype bank.
DEFAULT_PROTOS = "weights/detecty_prototypes.npz"
DEFAULT_PROTOTYPES_DIR = "third_party/detecty/prototypes"
DEFAULT_CATALOG_DIR = "third_party/detecty/objects_gt"

# Lazy import to avoid loading torch/transformers/timm until first use.
_detecty_imported = False


def _ensure_detecty() -> None:
    global _detecty_imported
    if _detecty_imported:
        return
    try:
        import detecty  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "detecty provider requires the detecty package. Install with: "
            "uv sync (it is an editable path dep on third_party/detecty)."
        ) from e
    _detecty_imported = True


def _resolve(path: str) -> str:
    """Resolve a config path against the repo root when it is relative."""
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else _REPO_ROOT / p)


class DetectyObjectDetectionProvider(ObjectDetectionProvider):
    """Object detection via the custom detecty ensemble (30 Incheon2026 objects)."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the detecty provider.

        Args:
            config: Optional keys:
                - device: "cuda" or "cpu" (default: auto — cuda if available).
                - use_ocr: EasyOCR brand-text matching (default: True). Best for
                  the brand twins; disable to skip the easyocr dependency at run
                  time.
                - use_vlm: Optional Gemma/vLLM consult on the hardest crops
                  (default: False; needs a served endpoint — see detecty docs).
                - quantize_localizer: INT8-quantize Grounding DINO, CPU-only
                  (default: False).
                - dino_model: DINOv3 backbone name (default: detecty's
                  vit_large_patch16_dinov3.lvd1689m).
                - protos: Prototype-bank cache path (default
                  "weights/detecty_prototypes.npz"). Built on first load from the
                  reference images below if missing.
                - prototypes_dir: In-domain crops dir (default
                  "third_party/detecty/prototypes").
                - catalog_dir: Official catalog photos dir (default
                  "third_party/detecty/objects_gt").
                - min_box: Minimum crop side in pixels (default: 8).
                - max_objects: Maximum detections to return (default: 50).
                - crop_padding: Pixels added around each returned bbox (default: 10).
                - min_area_ratio: Drop boxes smaller than this (default: 0.0005).
                - max_area_ratio: Drop boxes larger than this (default: 0.95).
                - include_review: Keep low-confidence "review" detections
                  (default: True). Set False to drop them.
        """
        self._config = config

        device = config.get("device")
        if not device:  # None or "" (config.toml uses "" for auto) -> autodetect
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._use_ocr = bool(config.get("use_ocr", True))
        self._use_vlm = bool(config.get("use_vlm", False))
        self._quantize = bool(config.get("quantize_localizer", False))
        self._dino_model = config.get("dino_model")  # None -> detecty default

        self._protos_path = _resolve(str(config.get("protos", DEFAULT_PROTOS)))
        self._prototypes_dir = _resolve(
            str(config.get("prototypes_dir", DEFAULT_PROTOTYPES_DIR))
        )
        self._catalog_dir = _resolve(
            str(config.get("catalog_dir", DEFAULT_CATALOG_DIR))
        )

        self._min_box = int(config.get("min_box", 8))
        self._max_objects = int(config.get("max_objects", 50))
        self._crop_padding = int(config.get("crop_padding", 10))
        self._min_area_ratio = float(config.get("min_area_ratio", 0.0005))
        self._max_area_ratio = float(config.get("max_area_ratio", 0.95))
        self._include_review = bool(config.get("include_review", True))

        self._sy: Any = None
        self._model_name = "detecty"
        # Set once we warn that detecty can't produce masks, so return_mask=True
        # prints the warning a single time rather than on every request.
        self._warned_no_masks = False

    def load_model(self) -> None:
        """Pre-load detecty (build the prototype bank if needed, then setup)."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Lazy-load detecty on first use: build prototype bank, load models."""
        if self._sy is not None:
            return
        _ensure_detecty()

        # Build the prototype bank once (DINOv3 embeddings + hue histograms of the
        # bundled reference images), then reuse the cached .npz on later starts.
        if not os.path.isfile(self._protos_path):
            os.makedirs(os.path.dirname(self._protos_path) or ".", exist_ok=True)
            from detecty.build_prototypes import main as build_protos

            argv = [
                "--prototypes-dir", self._prototypes_dir,
                "--catalog-dir", self._catalog_dir,
                "--out", self._protos_path,
                "--device", self._device,
            ]
            if self._dino_model:
                argv += ["--model", self._dino_model]
            build_protos(argv)

        from detecty import SamYolo

        kwargs: dict[str, Any] = dict(
            protos=self._protos_path,
            device=self._device,
            use_ocr=self._use_ocr,
            use_vlm=self._use_vlm,
            quantize_localizer=self._quantize,
        )
        if self._dino_model:
            kwargs["dino_model"] = self._dino_model
        self._sy = SamYolo(**kwargs)
        self._sy.setup()

    def detect(
        self,
        image: Image.Image,
        prompts: list[str] | None = None,
        return_mask: bool = False,
    ) -> list[DetectedObject]:
        """Run detecty inference and return a DetectedObject list.

        ``prompts`` is accepted for interface compatibility with the concept
        providers (e.g. SAM3/YOLOE) but ignored — detecty uses its fixed 30-class
        Incheon2026 taxonomy.

        ``return_mask`` cannot be honored: detecty localizes with Grounding DINO
        and has no segmentation head. A warning is printed once and detections
        are returned with a blank (``None``) mask.
        """
        self._ensure_loaded()
        assert self._sy is not None
        if return_mask:
            self._warn_no_masks_once()

        res = self._sy.detect(image, min_box=self._min_box)
        w = int(res.get("width", 0)) or image.width
        h = int(res.get("height", 0)) or image.height
        total_area = float(w * h) or 1.0
        pad = self._crop_padding

        # Order by area descending (prefer larger objects), like the other
        # providers, so max_objects keeps the most salient detections.
        raw = res.get("detections", [])
        raw_sorted = sorted(
            raw,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )

        detections: list[DetectedObject] = []
        for d in raw_sorted:
            if len(detections) >= self._max_objects:
                break
            if d.get("review") and not self._include_review:
                continue

            x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
            area_ratio = (x2 - x1) * (y2 - y1) / total_area
            if area_ratio < self._min_area_ratio or area_ratio > self._max_area_ratio:
                continue

            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)
            if x2p <= x1p or y2p <= y1p:
                continue

            class_id = d.get("class_id")
            detections.append(
                DetectedObject(
                    mask=None,
                    bbox=(x1p, y1p, x2p, y2p),
                    area_ratio=area_ratio,
                    class_id=int(class_id) if class_id is not None else None,
                    class_name=d.get("class"),
                    confidence=(
                        float(d["score"]) if d.get("score") is not None else None
                    ),
                )
            )
        return detections

    def _warn_no_masks_once(self) -> None:
        """Print a single warning when masks are requested but unavailable."""
        if self._warned_no_masks:
            return
        print(
            "detecty provider: model has no segmentation head; return_mask=True "
            "ignored, returning bounding boxes only."
        )
        self._warned_no_masks = True

    def get_model_name(self) -> str:
        return self._model_name
