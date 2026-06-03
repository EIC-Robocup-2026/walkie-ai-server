"""Base classes and data structures for face recognition.

The server-side face service is **stateless**: it turns a frame into a list of
detected faces, each carrying a bounding box, an L2-normalized embedding, and a
detection score. Enrollment, names, and matching all live in the agent repo —
this module never stores or compares faces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class FaceEmbedding:
    """A single detected face together with its recognition embedding."""

    bbox_xyxy: tuple[int, int, int, int]
    """Bounding box in ``(x1, y1, x2, y2)`` pixel coords (top-left, bottom-right).

    Explicitly ``xyxy`` — the agent depends on this and the rest of the
    ecosystem mixes ``xyxy``/``cxcywh``, so the convention is pinned here."""

    embedding: list[float] = field(default_factory=list)
    """L2-normalized recognition vector (``‖v‖₂ = 1``), constant dimension for
    every face and every call. The agent matches with cosine distance and never
    re-normalizes."""

    det_score: float = 0.0
    """Face-detection confidence in ``[0, 1]``."""


class FaceRecognitionProvider(ABC):
    """Abstract base class for face recognition providers."""

    @abstractmethod
    def embed(self, image: Image.Image) -> list[FaceEmbedding]:
        """Detect every face in *image* and return one embedding per face.

        Args:
            image: PIL Image (RGB) to process.

        Returns:
            List of ``FaceEmbedding`` instances, or ``[]`` when no face is found.
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
