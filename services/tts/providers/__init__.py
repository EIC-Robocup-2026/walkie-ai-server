"""TTS provider registry.

To add a new provider:
1. Create a new file in this directory (e.g., google.py)
2. Implement TTSProvider base class
3. Register it in PROVIDERS dict below
"""

from typing import Any

from ..base import TTSProvider
from .elevenlabs import ElevenLabsProvider
from .piper_tts import PiperProvider

# Provider registry - add new providers here
PROVIDERS: dict[str, type[TTSProvider]] = {
    "elevenlabs": ElevenLabsProvider,
    "piper": PiperProvider,
}


def get_provider(name: str, config: dict[str, Any]) -> TTSProvider:
    """Get a TTS provider instance by name.
    
    Args:
        name: Provider name (must be registered in PROVIDERS).
        config: Provider-specific configuration.
        
    Returns:
        Configured TTSProvider instance.
        
    Raises:
        ValueError: If provider is not registered.
    """
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown TTS provider: '{name}'. Available: {available}")
    
    return provider_class(config)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDERS.keys())
