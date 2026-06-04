"""Face recognition provider registry."""

from typing import Any

from ..base import FaceRecognitionProvider
from .insightface_provider import InsightFaceProvider

PROVIDERS: dict[str, type[FaceRecognitionProvider]] = {
    "insightface": InsightFaceProvider,
}


def get_provider(name: str, config: dict[str, Any]) -> FaceRecognitionProvider:
    """Get a face recognition provider instance by name."""
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown face recognition provider: '{name}'. Available: {available}"
        )
    return provider_class(config)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDERS.keys())
