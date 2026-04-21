"""Base class for Speech-to-Text providers."""

from abc import ABC, abstractmethod


class STTProvider(ABC):
    """Abstract base class for STT providers.
    
    Implement this class to add a new STT provider.
    """

    @abstractmethod
    def transcribe(self, audio_content: bytes, **kwargs) -> str:
        """Transcribe audio to text.
        
        Args:
            audio_content: Audio data as bytes.
            **kwargs: Provider-specific options.
            
        Returns:
            Transcribed text.
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes.
        
        Returns:
            List of language codes (e.g., ['en-US', 'es-ES']).
        """
        pass
