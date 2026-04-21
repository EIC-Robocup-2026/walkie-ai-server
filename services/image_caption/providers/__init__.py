"""Image Captioning provider registry.

To add a new provider:
1. Create a new file in this directory (e.g., openai.py)
2. Implement ImageCaptionProvider base class
3. Register it in PROVIDERS dict below
"""

from typing import Any

from ..base import ImageCaptionProvider
from .florence2_large import Florence2LargeImageCaptionProvider
from .google_caption import GoogleImageCaptionProvider
from .paligemma import PaliGemmaImageCaptionProvider

# Provider registry - add new providers here
PROVIDERS: dict[str, type[ImageCaptionProvider]] = {
    "google": GoogleImageCaptionProvider,
    "paligemma": PaliGemmaImageCaptionProvider,
    "florence2": Florence2LargeImageCaptionProvider,
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
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown image captioning provider: '{name}'. Available: {available}")
    
    return provider_class(config)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDERS.keys())
