"""Speech-to-Text module with pluggable providers."""

from .base import STTProvider
from .stt import STT

__all__ = ["STT", "STTProvider", "Microphone", "list_audio_devices", "print_audio_devices"]
