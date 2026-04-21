"""Base class for Text-to-Speech providers."""

from abc import ABC, abstractmethod
from typing import Iterator


class TTSProvider(ABC):
    """Abstract base class for TTS providers.
    
    Implement this class to add a new TTS provider.
    """

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio.
        
        Args:
            text: The text to convert to speech.
            
        Returns:
            Audio data as bytes.
        """
        pass

    @abstractmethod
    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Synthesize text to audio with streaming.
        
        Args:
            text: The text to convert to speech.
            
        Yields:
            Chunks of audio data as bytes.
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Get list of supported output formats.
        
        Returns:
            List of format identifiers (e.g., ['mp3', 'wav', 'pcm']).
        """
        pass

    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming synthesis.
        
        Returns:
            True if streaming is supported, False otherwise.
        """
        return True
