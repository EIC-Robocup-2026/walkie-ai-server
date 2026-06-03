"""Face recognition module with configurable providers.

Usage:
    from services.face_recognition import FaceRecognition

    fr = FaceRecognition(provider="insightface")
    fr.load_model()
    faces = fr.embed(pil_image)   # -> list[FaceEmbedding]
"""

from typing import Any

from PIL import Image

from .base import FaceEmbedding, FaceRecognitionProvider
from .providers import get_provider, list_providers


class FaceRecognition:
    """Face recognition interface with configurable providers."""

    def __init__(self, provider: str = "insightface", **provider_config: Any) -> None:
        """Initialize FaceRecognition with a provider.

        Args:
            provider: Provider name (e.g., ``"insightface"``).
            **provider_config: Provider-specific configuration.
        """
        self._provider_name = provider
        self._provider: FaceRecognitionProvider = get_provider(
            provider, provider_config
        )

    @property
    def provider_name(self) -> str:
        """Current provider name."""
        return self._provider_name

    @property
    def provider(self) -> FaceRecognitionProvider:
        """Underlying provider instance."""
        return self._provider

    def load_model(self) -> None:
        """Pre-load the provider's model weights into memory."""
        self._provider.load_model()

    def embed(self, image: Image.Image) -> list[FaceEmbedding]:
        """Detect faces in *image* and return one embedding per face."""
        return self._provider.embed(image)

    def get_model_name(self) -> str:
        """Model name for logging / vector provenance."""
        return self._provider.get_model_name()

    def get_embedding_dim(self) -> int:
        """Dimension of the embedding vectors."""
        return self._provider.get_embedding_dim()

    @staticmethod
    def available_providers() -> list[str]:
        """List all available face recognition providers."""
        return list_providers()
