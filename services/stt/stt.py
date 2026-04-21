"""Speech-to-Text module with configurable providers.

Usage:
    from src.audio import STT
    
    # Initialize with default provider (google)
    stt = STT()
    
    # Or with explicit config
    stt = STT(provider="google", language_codes=["en-US", "th-TH"])
    
    # Transcribe audio
    text = stt.transcribe(audio_bytes)
"""

from .base import STTProvider
from .providers import get_provider, list_providers


class STT:
    """Speech-to-Text interface with configurable providers."""

    def __init__(
        self,
        provider: str = "google",
        **provider_config,
    ) -> None:
        """Initialize STT with a provider.

        Args:
            provider: The name of the provider to use (e.g., "google").
            **provider_config: Provider-specific configuration passed directly.
        """
        self._provider_name = provider
        self._provider: STTProvider = get_provider(provider, provider_config)

    @property
    def provider_name(self) -> str:
        """Get the current provider name."""
        return self._provider_name

    @property
    def provider(self) -> STTProvider:
        """Get the underlying provider instance."""
        return self._provider

    def transcribe(self, audio_content: bytes, **kwargs) -> str:
        """Transcribe audio to text."""
        return self._provider.transcribe(audio_content, **kwargs)

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages for current provider."""
        return self._provider.get_supported_languages()

    @staticmethod
    def available_providers() -> list[str]:
        """List all available STT providers."""
        return list_providers()
