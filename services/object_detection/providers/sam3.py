"""SAM3 object detection provider (open-vocabulary concept segmentation).

SAM 3 (Meta, 2025) performs *promptable concept segmentation*: given short
open-vocabulary noun-phrase text prompts it finds and segments every instance
of each concept. Unlike YOLO (fixed 365-class detector, no masks) this provider
returns segmentation masks and is not limited to a fixed label set, which makes
it a drop-in replacement for the YOLO provider when grounding/segmentation is
needed.

Used via the Ultralytics ``SAM3SemanticPredictor`` (text prompts). The
``sam3.pt`` weights must be downloaded manually from Hugging Face — they do not
auto-download. Set the ``model`` config to the local checkpoint path.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

from ..base import DetectedObject, ObjectDetectionProvider

# Default open-vocabulary concepts used when no per-request prompts are given.
# This is the RoboCup@Home known-object set (GermanOpen 2026 / 2026-season
# rulebook), grouped by the official Object Categories. Override via the
# ``prompts`` config key or per-request prompts.
# Source: https://github.com/RoboCupAtHome/GermanOpen2026 (objects/known_objects)
_DEFAULT_PROMPTS: tuple[str, ...] = (
    # cleaning supplies
    "cloth",
    "sponge",
    # dishes
    "bowl",
    "cup",
    "fork",
    "knife",
    "plate",
    "spoon",
    # drinks
    "coffee creamer",
    "coke",
    "ice tea",
    "milk",
    "orange juice",
    "red bull",
    "water bottle",
    # foods
    "bread",
    "cornflakes",
    "instant noodles",
    "potato",
    "tomato soup",
    # fruits
    "apple",
    "avocado",
    "lemon",
    "orange",
    # snacks
    "chips",
    "cookies",
    "gum",
    "mixed nuts",
    "pringles",
    # toiletries
    "hand creme",
    "soap",
    "toothpaste",
    # furniture / scene
    "large shelf",
    "dish rack",
    "shelf",
    "monitor",
    "bed",
    "pillow",
)

# Lazy imports to avoid loading torch/ultralytics until first use.
_ultralytics_imported = False


def _ensure_ultralytics() -> None:
    global _ultralytics_imported
    if _ultralytics_imported:
        return
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "SAM3 provider requires ultralytics (with SAM3 support). "
            "Install/upgrade with: pip install -U ultralytics"
        ) from e
    _ultralytics_imported = True


def _resolve_model_path(config: dict[str, Any]) -> str:
    """Resolve the SAM3 checkpoint: local file, HF download, or model name.

    SAM3 weights are gated and do not auto-download. Prefer a local path via the
    ``model`` config key. If an ``hf_repo`` is given we try to fetch it.
    """
    model = config.get("model", "sam3.pt")
    if os.path.isfile(model):
        return model

    hf_repo = config.get("hf_repo")
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download

            filename = config.get("hf_filename", os.path.basename(model) or "sam3.pt")
            return hf_hub_download(repo_id=hf_repo, filename=filename)
        except Exception as e:  # pragma: no cover - network/credential dependent
            raise FileNotFoundError(
                f"Could not download SAM3 weights '{model}' from '{hf_repo}'. "
                "Ensure access is granted and HF_TOKEN is set. "
                f"Error: {e}"
            ) from e

    # Pass through (e.g. an Ultralytics-resolvable name). Ultralytics will raise
    # a clear error if the gated weights are missing locally.
    return model


class SAM3ObjectDetectionProvider(ObjectDetectionProvider):
    """Open-vocabulary detection + segmentation via Meta SAM 3."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize SAM3 provider.

        Args:
            config: Optional keys:
                - model: Path to ``sam3.pt`` (default: "sam3.pt"). Weights must
                  be downloaded manually; or set ``hf_repo``/``hf_filename``.
                - hf_repo / hf_filename: Hugging Face repo to download weights.
                - device: "cuda" or "cpu" (default: auto).
                - half: Use FP16 inference (default: True on CUDA).
                - prompts: Default text concepts (list[str]) to detect when no
                  per-request prompts are supplied.
                - conf_threshold: Minimum confidence (0-1) to keep (default: 0.25).
                - max_objects: Maximum detections to return (default: 50).
                - crop_padding: Pixels added around each crop bbox (default: 10).
                - min_area_ratio: Drop boxes smaller than this (default: 0.0005).
                - max_area_ratio: Drop boxes larger than this (default: 0.95).
                - imgsz: Inference image size (default: 640). The SAM3 backbone
                  cost (and GPU memory) scales ~quadratically with this; lower
                  it (e.g. 512) for speed at the cost of small-object recall,
                  or raise it for recall. NOTE: the native 1024 OOMs on a 24 GB
                  GPU with the full default prompt set. Set to ``None`` to use
                  the model default (1024).
                - compile: torch.compile() the model for faster inference
                  (default: False). True / "default" / "reduce-overhead" /
                  "max-autotune-no-cudagraphs". Adds a one-time compile cost on
                  the first run, absorbed by ``load_model``'s warm-up.
                - warmup: Run a dummy inference in ``load_model`` to pay model
                  build / compile / CUDA-kernel autotune cost up front instead
                  of on the first real request (default: True).
        """
        self._config = config

        device = config.get("device")
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SAM3 provider using device: {device or 'auto'}")
        self._device = device
        self._half = bool(config.get("half", device == "cuda"))

        prompts = config.get("prompts")
        self._default_prompts: list[str] = (
            list(prompts) if prompts else list(_DEFAULT_PROMPTS)
        )

        self._conf_threshold = float(config.get("conf_threshold", 0.25))
        self._max_objects = int(config.get("max_objects", 50))
        self._crop_padding = int(config.get("crop_padding", 10))
        self._min_area_ratio = float(config.get("min_area_ratio", 0.0005))
        self._max_area_ratio = float(config.get("max_area_ratio", 0.95))
        # Default below the model's native 1024 for speed; ``None`` opts back
        # into the model default.
        self._imgsz = config.get("imgsz", 640)
        self._compile = config.get("compile", False)
        self._warmup = bool(config.get("warmup", True))

        self._predictor = None
        self._model_name = os.path.basename(str(config.get("model", "sam3.pt")))

    def load_model(self) -> None:
        """Pre-load SAM3 model weights into memory (and warm up if enabled)."""
        self._ensure_loaded()
        if self._warmup:
            self._run_warmup()

    def _run_warmup(self) -> None:
        """Pay model-build / torch.compile / CUDA-autotune cost up front.

        Runs one dummy inference so the first real request isn't hit with the
        one-time compile/kernel-autotune latency (especially when ``compile`` is
        enabled). Failures here are non-fatal — the real call will retry.
        """
        assert self._predictor is not None
        size = int(self._imgsz) if self._imgsz else 1024
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        try:
            self._predictor.set_image(dummy)
            self._predictor(text=self._default_prompts[:1] or ["object"])
            self._predictor.reset_image()
            print(f"SAM3 provider warmed up (imgsz={size}, compile={self._compile}).")
        except Exception as e:  # pragma: no cover - warm-up is best-effort
            print(f"SAM3 warm-up skipped: {e}")

    def _ensure_loaded(self) -> None:
        """Lazy-load the SAM3 semantic predictor on first use."""
        if self._predictor is not None:
            return
        _ensure_ultralytics()
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides: dict[str, Any] = dict(
            conf=self._conf_threshold,
            task="segment",
            mode="predict",
            model=_resolve_model_path(self._config),
            device=self._device,
            half=self._half,
            save=False,
            verbose=False,
        )
        if self._imgsz is not None:
            overrides["imgsz"] = int(self._imgsz)
        if self._compile:
            # True / "default" / "reduce-overhead" / "max-autotune-no-cudagraphs"
            overrides["compile"] = self._compile
        self._predictor = SAM3SemanticPredictor(overrides=overrides)

    def detect(
        self,
        image: Image.Image,
        prompts: list[str] | None = None,
    ) -> list[DetectedObject]:
        """Run SAM3 concept segmentation and return DetectedObject list.

        Args:
            image: PIL Image (RGB).
            prompts: Open-vocabulary noun phrases to find (e.g.
                ["red mug", "cereal box"]). Falls back to the configured default
                prompts when omitted.
        """
        self._ensure_loaded()
        assert self._predictor is not None

        concepts = [p for p in (prompts or self._default_prompts) if p and p.strip()]
        if not concepts:
            return []

        img_rgb = np.array(image)
        if img_rgb.ndim == 2:
            img_rgb = np.stack([img_rgb] * 3, axis=-1)
        h, w = img_rgb.shape[0], img_rgb.shape[1]
        total_area = float(h * w)

        self._predictor.set_image(img_rgb)
        results = self._predictor(text=concepts)

        return self._parse_results(results, concepts, w, h, total_area)

    def _parse_results(
        self,
        results: Any,
        concepts: list[str],
        w: int,
        h: int,
        total_area: float,
    ) -> list[DetectedObject]:
        """Convert Ultralytics SAM3 results into DetectedObject list."""
        if not results:
            return []
        r = results[0] if isinstance(results, (list, tuple)) else results

        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)
        if boxes is None and masks is None:
            return []

        xyxy = conf = cls = None
        if boxes is not None and len(boxes) > 0:
            # .float() first: with half=True the boxes are float16, whose max
            # (~65504) overflows when computing pixel areas and rounds integer
            # coords above 2048. float32 is exact for image-space coordinates.
            xyxy = boxes.xyxy.float().cpu().numpy()
            conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            cls = boxes.cls.cpu().numpy() if boxes.cls is not None else None

        mask_data = None
        if masks is not None and getattr(masks, "data", None) is not None:
            mask_data = masks.data.cpu().numpy()  # (N, mh, mw), float/bool

        # Number of detections from whichever source is present.
        n = 0
        if xyxy is not None:
            n = len(xyxy)
        elif mask_data is not None:
            n = mask_data.shape[0]
        if n == 0:
            return []

        names = getattr(r, "names", None)

        # Order by area descending (prefer larger objects), like YOLO provider.
        if xyxy is not None:
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        else:
            areas = np.array([float(m.sum()) for m in mask_data])
        order = np.argsort(-areas)
        pad = self._crop_padding

        detections: list[DetectedObject] = []
        for idx in order:
            if len(detections) >= self._max_objects:
                break

            mask_2d = None
            if mask_data is not None and idx < mask_data.shape[0]:
                mask_2d = self._resize_mask(mask_data[idx], w, h)

            bbox = self._bbox_for(xyxy, idx, mask_2d, w, h)
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
            class_name = self._label_for(cls, idx, names, concepts)
            class_id = int(cls[idx]) if cls is not None else None

            detections.append(
                DetectedObject(
                    mask=mask_2d,
                    bbox=[x1p, y1p, x2p, y2p],
                    area_ratio=area_ratio,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                )
            )
        return detections

    @staticmethod
    def _resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
        """Binarize and resize a mask to the original image size (uint8 H×W)."""
        binary = (np.asarray(mask) > 0.5).astype(np.uint8)
        if binary.shape == (h, w):
            return binary
        resized = Image.fromarray(binary * 255).resize((w, h), Image.NEAREST)
        return (np.array(resized) > 127).astype(np.uint8)

    @staticmethod
    def _bbox_for(
        xyxy: np.ndarray | None,
        idx: int,
        mask_2d: np.ndarray | None,
        w: int,
        h: int,
    ) -> tuple[int, int, int, int] | None:
        """Get an integer (x1, y1, x2, y2) bbox from boxes or mask extent."""
        if xyxy is not None and idx < len(xyxy):
            x1, y1, x2, y2 = xyxy[idx]
            return int(x1), int(y1), int(x2), int(y2)
        if mask_2d is not None:
            ys, xs = np.nonzero(mask_2d)
            if xs.size == 0 or ys.size == 0:
                return None
            return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        return None

    @staticmethod
    def _label_for(
        cls: np.ndarray | None,
        idx: int,
        names: Any,
        concepts: list[str],
    ) -> str:
        """Map a detection to its concept label."""
        if cls is not None and idx < len(cls):
            cid = int(cls[idx])
            if isinstance(names, dict) and cid in names:
                return str(names[cid])
            if isinstance(names, (list, tuple)) and 0 <= cid < len(names):
                return str(names[cid])
            if 0 <= cid < len(concepts):
                return concepts[cid]
        # Single-concept queries: label with that concept.
        if len(concepts) == 1:
            return concepts[0]
        return "object"

    def get_model_name(self) -> str:
        return self._model_name
