"""Embedding provider registry.

To add a new provider:
1. Create a new file in this directory (e.g., openai.py)
2. Implement EmbeddingProvider base class
3. Register it in PROVIDERS dict below
"""

from typing import Any

from ..base import EmbeddingProvider
from .clip import CLIPEmbeddingProvider

PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "clip": CLIPEmbeddingProvider,
}


def get_provider(name: str, config: dict[str, Any]) -> EmbeddingProvider:
    """Get an embedding provider instance by name."""
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown embedding provider: '{name}'. Available: {available}"
        )
    return provider_class(config)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDERS.keys())
