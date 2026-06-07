"""Object detection module with configurable providers.

Usage:
    from src.vision.object_detection import ObjectDetection

    det = ObjectDetection(provider="sam")
    objects = det.detect(pil_image)
"""

from typing import Any

from PIL import Image

from .base import DetectedObject, ObjectDetectionProvider
from .providers import get_provider, list_providers


class ObjectDetection:
    """Object detection interface with configurable providers."""

    def __init__(self, provider: str = "sam", **provider_config: Any) -> None:
        """Initialize ObjectDetection with a provider.

        Args:
            provider: Provider name (e.g., "sam").
            **provider_config: Provider-specific configuration.
        """
        self._provider_name = provider
        self._provider: ObjectDetectionProvider = get_provider(
            provider, provider_config
        )

    @property
    def provider_name(self) -> str:
        """Current provider name."""
        return self._provider_name

    @property
    def provider(self) -> ObjectDetectionProvider:
        """Underlying provider instance."""
        return self._provider

    def load_model(self) -> None:
        """Pre-load the provider's model weights into memory."""
        self._provider.load_model()

    def detect(
        self,
        image: Image.Image,
        prompts: list[str] | None = None,
        return_mask: bool = False,
    ) -> list[DetectedObject]:
        """Detect objects in an image.

        Args:
            image: PIL Image (RGB) to process.
            prompts: Optional text prompts for concept-segmentation providers
                (e.g. SAM3, YOLOE). Ignored by fixed-class providers (e.g. YOLO).
            return_mask: When True, populate ``DetectedObject.mask`` with a
                segmentation mask where the provider/model supports it.
        """
        return self._provider.detect(image, prompts, return_mask)

    def get_model_name(self) -> str:
        """Model name for logging."""
        return self._provider.get_model_name()

    @staticmethod
    def available_providers() -> list[str]:
        """List all available object detection providers."""
        return list_providers()
