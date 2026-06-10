"""Appearance (attire) re-ID embedding module with configurable providers.

Usage:
    from services.appearance import Appearance

    ap = Appearance(provider="osnet")
    ap.load_model()
    emb = ap.embed(person_crop)   # -> list[float], L2-normalized
"""

from typing import Any

from PIL import Image

from .base import AppearanceProvider
from .providers import get_provider, list_providers


class Appearance:
    """Appearance embedding interface with configurable providers."""

    def __init__(self, provider: str = "osnet", **provider_config: Any) -> None:
        """Initialize Appearance with a provider.

        Args:
            provider: Provider name (e.g., ``"osnet"``).
            **provider_config: Provider-specific configuration.
        """
        self._provider_name = provider
        self._provider: AppearanceProvider = get_provider(provider, provider_config)

    @property
    def provider_name(self) -> str:
        """Current provider name."""
        return self._provider_name

    @property
    def provider(self) -> AppearanceProvider:
        """Underlying provider instance."""
        return self._provider

    def load_model(self) -> None:
        """Pre-load the provider's model weights into memory."""
        self._provider.load_model()

    def embed(self, image: Image.Image) -> list[float]:
        """Embed a person crop into one L2-normalized appearance vector."""
        return self._provider.embed(image)

    def get_model_name(self) -> str:
        """Model name for logging / vector provenance."""
        return self._provider.get_model_name()

    def get_embedding_dim(self) -> int:
        """Dimension of the embedding vectors."""
        return self._provider.get_embedding_dim()

    @staticmethod
    def available_providers() -> list[str]:
        """List all available appearance providers."""
        return list_providers()
