"""Base class for Object Detection providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class DetectedObject:
    """A single detected object from an image."""

    mask: "np.ndarray | None"  # 2D uint8 (H, W) {0,1} segmentation mask, or None
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_ratio: float  # fraction of image area
    # Optional: set by providers that output class and confidence (e.g. YOLO)
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = None


class ObjectDetectionProvider(ABC):
    """Abstract base class for object detection/segmentation providers."""

    @abstractmethod
    def detect(
        self,
        image: Image.Image,
        prompts: list[str] | None = None,
        return_mask: bool = False,
    ) -> list[DetectedObject]:
        """Detect and segment objects in an image.

        Args:
            image: PIL Image (RGB) to process.
            prompts: Optional open-vocabulary text prompts (noun phrases) to
                look for. Used by concept-segmentation providers such as SAM3
                and YOLOE; ignored by fixed-class providers such as YOLO.
            return_mask: When True, populate each ``DetectedObject.mask`` with a
                segmentation mask (if the provider/model supports it). When
                False (default) masks are omitted (``mask=None``) and only
                bounding boxes are returned. Providers/models that cannot
                segment print a warning and return blank (``None``) masks.

        Returns:
            List of DetectedObject with bbox, area_ratio (and mask when
            ``return_mask`` is set and supported).
        """
        pass

    def load_model(self) -> None:
        """Pre-load model weights into memory.

        Default implementation is a no-op. Override in providers that use
        lazy loading so that ``load_model()`` can be called eagerly.
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return a short model name for logging."""
        pass


# ---------------------------------------------------------------------------
# Shared parsing helpers (used by the YOLO, SAM3 and YOLOE providers).
# ---------------------------------------------------------------------------


def resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """Binarize and resize a mask to the original image size (uint8 H×W {0,1})."""
    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    if binary.shape == (h, w):
        return binary
    resized = Image.fromarray(binary * 255).resize((w, h), Image.NEAREST)
    return (np.array(resized) > 127).astype(np.uint8)


def bbox_for(
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


def label_for(
    cls: np.ndarray | None,
    idx: int,
    names: Any,
    concepts: list[str],
) -> str:
    """Map a detection to its concept/class label."""
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
