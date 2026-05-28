"""Whisper STT provider implementation using faster-whisper."""

import logging
from typing import Any

import numpy as np
import torch

from ..base import STTProvider

logger = logging.getLogger(__name__)


class WhisperSTTProvider(STTProvider):
    """Whisper Speech-to-Text provider using faster-whisper for local inference.
    
    Uses faster-whisper for efficient on-device inference, supporting both
    CUDA and CPU execution with various quantization options.
    """

    # Whisper supports 99 languages
    SUPPORTED_LANGUAGES = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
        "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
        "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
        "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
        "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
        "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    ]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Whisper STT provider.
        
        Args:
            config: Provider configuration with optional keys:
                - model_name: Model size (default: "small")
                  Options: "tiny", "base", "small", "medium", "large-v3"
                - device: Compute device "cuda" or "cpu"
                  (default: "cuda" if available, else "cpu")
                - language: Target language code (default: "en")
                - compute_type: Quantization "float16", "int8", "float32" (default: "float16")
                - beam_size: Beam size for decoding (default: 5)
                - vad_filter: Use VAD filtering (default: True)
        """
        self.model_name = config.get("model_name", "small")
        device = config.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.language = config.get("language", "en")
        self.compute_type = config.get("compute_type", "float16")
        self.beam_size = config.get("beam_size", 5)
        self.vad_filter = config.get("vad_filter", True)
        
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model, falling back to CPU if CUDA is unavailable."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required for Whisper provider. "
                "Install with: pip install faster-whisper"
            )

        logger.info(
            f"Loading Whisper model: {self.model_name} "
            f"(device={self.device}, compute_type={self.compute_type})"
        )

        try:
            self.model = self._build_model(WhisperModel, self.device, self.compute_type)
        except Exception as e:
            # CUDA may be requested but unavailable, or have a driver/runtime
            # mismatch; fall back to CPU so the server can still start.
            if self.device != "cpu":
                logger.warning(
                    f"Failed to load Whisper on '{self.device}' ({e}); "
                    "falling back to CPU (compute_type=int8)..."
                )
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = self._build_model(WhisperModel, "cpu", "int8")
            else:
                logger.error(f"Failed to load Whisper model: {e}")
                raise

        logger.info(
            f"Whisper model loaded: {self.model_name} "
            f"(device={self.device}, compute_type={self.compute_type})"
        )

    def _build_model(self, whisper_model_cls, device: str, compute_type: str):
        """Instantiate WhisperModel, downgrading float16 to int8 if unsupported."""
        try:
            return whisper_model_cls(
                self.model_name, device=device, compute_type=compute_type
            )
        except Exception as e:
            # Fall back to int8 if float16 not supported (e.g., on CPU / Mac)
            if "float16" in str(e).lower() or "float16" in compute_type:
                logger.warning(
                    f"compute_type '{compute_type}' not supported on {device}, "
                    "falling back to int8..."
                )
                self.compute_type = "int8"
                return whisper_model_cls(
                    self.model_name, device=device, compute_type="int8"
                )
            raise

    def transcribe(
        self,
        audio_content: bytes,
        prompt: str = "You are Walkie agent.",
        **kwargs,
    ) -> str:
        """Transcribe audio to text.
        
        Args:
            audio_content: Raw PCM audio (16-bit signed, 16kHz, mono).
            prompt: Prompt to use for transcription.
            **kwargs: Provider-specific options (unused).
        Returns:
            Transcribed text.
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        logger.debug(f"Transcribing {len(audio_content)} bytes of audio")
        
        # Convert 16-bit PCM bytes to normalized float32 array
        audio_array = np.frombuffer(audio_content, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Run inference
        segments, info = self.model.transcribe(
            audio_array,
            language=self.language,
            vad_filter=self.vad_filter,
            beam_size=self.beam_size,
            hotwords=prompt,
        )
        
        # Join all segment texts
        text = " ".join([segment.text for segment in segments]).strip()
        
        logger.debug(f"Transcribed: '{text}'")
        return text

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return self.SUPPORTED_LANGUAGES.copy()
