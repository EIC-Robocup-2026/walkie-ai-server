"""Text-to-Speech module with configurable providers.

Usage:
    from src.audio import TTS
    
    # Initialize with default provider (elevenlabs)
    tts = TTS()
    
    # Or with explicit config
    tts = TTS(provider="elevenlabs", voice_id="21m00Tcm4TlvDq8ikWAM")
    
    # Synthesize audio
    audio = tts.synthesize("Hello, world!")
    
    # Stream audio
    for chunk in tts.synthesize_stream("Hello, world!"):
        # process chunk
        pass
"""

from typing import Iterator

from .base import TTSProvider
from .providers import get_provider, list_providers


class TTS:
    """Text-to-Speech interface with configurable providers."""

    def __init__(
        self,
        provider: str = "elevenlabs",
        **provider_config,
    ) -> None:
        """Initialize TTS with a provider.
        
        Args:
            provider: The name of the provider to use (e.g., "elevenlabs").
            **provider_config: Provider-specific configuration passed directly.
        """
        self._provider_name = provider
        self._provider: TTSProvider = get_provider(provider, provider_config)

    @property
    def provider_name(self) -> str:
        """Get the current provider name."""
        return self._provider_name

    @property
    def provider(self) -> TTSProvider:
        """Get the underlying provider instance."""
        return self._provider

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio.
        
        Args:
            text: The text to convert to speech.
            
        Returns:
            Audio data as bytes.
        """
        return self._provider.synthesize(text)

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Synthesize text to audio with streaming.
        
        Args:
            text: The text to convert to speech.
            
        Yields:
            Chunks of audio data as bytes.
        """
        yield from self._provider.synthesize_stream(text)

    def get_supported_formats(self) -> list[str]:
        """Get list of supported output formats for current provider."""
        return self._provider.get_supported_formats()

    def supports_streaming(self) -> bool:
        """Check if current provider supports streaming."""
        return self._provider.supports_streaming()

    @staticmethod
    def available_providers() -> list[str]:
        """List all available TTS providers."""
        return list_providers()
