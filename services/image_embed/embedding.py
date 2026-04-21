"""Embedding module with configurable providers.

Usage:
    from src.vision.embedding import Embedding

    emb = Embedding(provider="clip")
    img_vec = emb.embed_image(pil_image)
    text_vec = emb.embed_text("a chair")
    sim = emb.similarity(img_vec, text_vec)
"""

from typing import Any

from PIL import Image

from .base import EmbeddingProvider
from .providers import get_provider, list_providers


class Embedding:
    """Embedding interface with configurable providers."""

    def __init__(self, provider: str = "clip", **provider_config: Any) -> None:
        """Initialize Embedding with a provider.

        Args:
            provider: Provider name (e.g., "clip").
            **provider_config: Provider-specific configuration.
        """
        self._provider_name = provider
        self._provider: EmbeddingProvider = get_provider(provider, provider_config)

    @property
    def provider_name(self) -> str:
        """Current provider name."""
        return self._provider_name

    @property
    def provider(self) -> EmbeddingProvider:
        """Underlying provider instance."""
        return self._provider

    def load_model(self) -> None:
        """Pre-load the provider's model weights into memory."""
        self._provider.load_model()

    def embed_image(self, image: Image.Image) -> list[float]:
        """Compute embedding for an image."""
        return self._provider.embed_image(image)

    def embed_text(self, text: str) -> list[float]:
        """Compute embedding for a single text."""
        return self._provider.embed_text(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for multiple texts."""
        return self._provider.embed_texts(texts)

    def similarity(self, embedding_a: list[float], embedding_b: list[float]) -> float:
        """Cosine similarity between two embeddings."""
        return self._provider.similarity(embedding_a, embedding_b)

    def get_embedding_dim(self) -> int:
        """Dimension of embedding vectors."""
        return self._provider.get_embedding_dim()

    @staticmethod
    def available_providers() -> list[str]:
        """List all available embedding providers."""
        return list_providers()
