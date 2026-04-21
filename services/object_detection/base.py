"""Base class for Object Detection providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass
class DetectedObject:
    """A single detected object from an image."""

    mask: "tuple[tuple[int, int], ...] | None"  # (y, x) coordinates or None
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_ratio: float  # fraction of image area
    cropped_image: Image.Image  # cropped region with padding
    # Optional: set by providers that output class and confidence (e.g. YOLO)
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = None


class ObjectDetectionProvider(ABC):
    """Abstract base class for object detection/segmentation providers."""

    @abstractmethod
    def detect(self, image: Image.Image) -> list[DetectedObject]:
        """Detect and segment objects in an image.

        Args:
            image: PIL Image (RGB) to process.

        Returns:
            List of DetectedObject with mask, bbox, area_ratio, cropped_image.
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
