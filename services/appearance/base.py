"""Base classes for the appearance (attire) re-ID embedding service.

The server-side appearance service is **stateless**: one person crop in →
one L2-normalized appearance embedding out. It is the second modality of the
agent's people memory — re-identifying a person whose face is *not* visible
(turned away, far, occluded). Enrollment, fusion scoring, thresholds, and the
people database all live in the agent repo; this module never stores or
compares embeddings.

Pipeline design by Chalk (EIC team), adopted from the ``eic-human``
subproject (``eic_human/pipeline/appearance.py``). Contract frozen in
``docs/appearance_service_handoff.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class AppearanceProvider(ABC):
    """Abstract base class for appearance embedding providers."""

    @abstractmethod
    def embed(self, image: Image.Image) -> list[float]:
        """Embed *image* (a person crop) into one appearance vector.

        The provider embeds **whatever image it is given** — no person
        detection or cropping happens here; the agent crops to the person
        bbox before sending.

        Args:
            image: PIL Image (RGB) of one person.

        Returns:
            L2-normalized embedding (``‖v‖₂ = 1``), constant dimension for
            every call. The agent matches with cosine similarity and never
            re-normalizes.
        """

    def load_model(self) -> None:
        """Pre-load model weights into memory.

        Default implementation is a no-op. Override in providers that use lazy
        loading so that ``load_model()`` can be called eagerly.
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return a short model name for logging / vector provenance."""

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the (constant) dimension of the embedding vectors."""
