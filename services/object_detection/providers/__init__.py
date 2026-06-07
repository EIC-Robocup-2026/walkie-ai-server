"""Object detection provider registry."""

from typing import Any

from ..base import ObjectDetectionProvider
from .sam3 import SAM3ObjectDetectionProvider
from .yolo import YOLOObjectDetectionProvider
from .yoloe import YOLOEObjectDetectionProvider

PROVIDERS: dict[str, type[ObjectDetectionProvider]] = {
    "yolo": YOLOObjectDetectionProvider,
    "sam3": SAM3ObjectDetectionProvider,
    "yoloe": YOLOEObjectDetectionProvider,
}


def get_provider(name: str, config: dict[str, Any]) -> ObjectDetectionProvider:
    """Get an object detection provider instance by name."""
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown object detection provider: '{name}'. Available: {available}"
        )
    return provider_class(config)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDERS.keys())
