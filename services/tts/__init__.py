"""Text-to-Speech module with pluggable providers."""

from .base import TTSProvider
from .tts import TTS

__all__ = ["TTS", "TTSProvider"]
