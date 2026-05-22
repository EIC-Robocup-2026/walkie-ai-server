"""Image Captioning provider registry.

To add a new provider:
1. Create a new file in this directory (e.g., openai.py)
2. Implement ImageCaptionProvider base class
3. Register it in PROVIDERS dict below
"""

from typing import Any, Callable

from ..base import ImageCaptionProvider
from .florence2 import Florence2ImageCaptionProvider
from .google_caption import GoogleImageCaptionProvider
from .paligemma import PaliGemmaImageCaptionProvider

ProviderFactory = Callable[[dict[str, Any]], ImageCaptionProvider]


def _florence2(variant: str) -> ProviderFactory:
    def factory(config: dict[str, Any]) -> ImageCaptionProvider:
        return Florence2ImageCaptionProvider({"variant": variant, **config})
    return factory


# Provider registry - add new providers here
PROVIDERS: dict[str, ProviderFactory] = {
    "google": GoogleImageCaptionProvider,
    "paligemma": PaliGemmaImageCaptionProvider,
    "florence2-base": _florence2("base"),
    "florence2-large": _florence2("large"),
}


def get_provider(name: str, config: dict[str, Any]) -> ImageCaptionProvider:
    """Get an image captioning provider instance by name.

    Args:
        name: Provider name (must be registered in PROVIDERS).
        config: Provider-specific configuration.

    Returns:
        Configured ImageCaptionProvider instance.

    Raises:
        ValueError: If provider is not registered.
    """
    factory = PROVIDERS.get(name)
    if factory is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown image captioning provider: '{name}'. Available: {available}")

    return factory(config)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDERS.keys())
