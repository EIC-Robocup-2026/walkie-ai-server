"""Embedding module with pluggable providers."""

from .base import EmbeddingProvider
from .embedding import Embedding

__all__ = ["Embedding", "EmbeddingProvider"]
