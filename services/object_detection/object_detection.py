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

    def detect(self, image: Image.Image) -> list[DetectedObject]:
        """Detect objects in an image."""
        return self._provider.detect(image)

    def get_model_name(self) -> str:
        """Model name for logging."""
        return self._provider.get_model_name()

    @staticmethod
    def available_providers() -> list[str]:
        """List all available object detection providers."""
        return list_providers()
