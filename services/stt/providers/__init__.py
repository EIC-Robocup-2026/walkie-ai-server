"""STT provider registry.

To add a new provider:
1. Create a new file in this directory (e.g., whisper.py)
2. Implement STTProvider base class
3. Register it in PROVIDERS dict below
"""

from typing import Any

from ..base import STTProvider
from .google import GoogleSTTProvider
from .whisper import WhisperSTTProvider

# Provider registry - add new providers here
PROVIDERS: dict[str, type[STTProvider]] = {
    "google": GoogleSTTProvider,
    "whisper": WhisperSTTProvider,
}


def get_provider(name: str, config: dict[str, Any]) -> STTProvider:
    """Get an STT provider instance by name."""
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown STT provider: '{name}'. Available: {available}")
    
    return provider_class(config)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDERS.keys())
