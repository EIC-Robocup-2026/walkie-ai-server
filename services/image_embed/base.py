"""Base class for Embedding providers."""

from abc import ABC, abstractmethod

from PIL import Image


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Implement this class to add a new embedding provider (e.g., CLIP).
    """

    @abstractmethod
    def embed_image(self, image: Image.Image) -> list[float]:
        """Compute embedding vector for an image.

        Args:
            image: PIL Image to embed.

        Returns:
            Normalized embedding as list of floats.
        """
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Compute embedding vector for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Normalized embedding as list of floats.
        """
        pass

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for multiple texts (batch).

        Default implementation calls embed_text for each. Override for batch efficiency.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of normalized embedding vectors.
        """
        return [self.embed_text(t) for t in texts]

    def similarity(self, embedding_a: list[float], embedding_b: list[float]) -> float:
        """Cosine similarity between two embeddings (assumed normalized).

        Args:
            embedding_a: First embedding vector.
            embedding_b: Second embedding vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        import math
        dot = sum(a * b for a, b in zip(embedding_a, embedding_b))
        return max(-1.0, min(1.0, dot))

    def load_model(self) -> None:
        """Pre-load model weights into memory.

        Default implementation is a no-op. Override in providers that use
        lazy loading so that ``load_model()`` can be called eagerly.
        """

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimension of embedding vectors."""
        pass
