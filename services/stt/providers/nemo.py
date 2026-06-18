"""NeMo STT provider — Chalk's local NemotronASR (FastConformer -> RNNT)."""

import logging
import os
from typing import Any

import numpy as np
import torch

from ..base import STTProvider

logger = logging.getLogger(__name__)

# WAV (RIFF) header length for 16 kHz/mono/16-bit PCM written by the standard 44-byte
# canonical header. The robot streams *raw* PCM (no header), but tests and some callers
# post a real .wav; strip the header so we don't feed the model ~22 garbage samples.
_WAV_HEADER_LEN = 44


class NemoSTTProvider(STTProvider):
    """Speech-to-Text via the local ``eic_speech_nemo`` NemotronASR model.

    Loads a single ``.pt`` checkpoint once at startup and reuses it for every
    request.  Expects the same audio contract as the other providers: raw 16-bit
    signed PCM, 16 kHz, mono (the robot's microphone already resamples to this).
    """

    # The checkpoint Chalk ships is trained for English; this is checkpoint-dependent.
    SUPPORTED_LANGUAGES = ["en"]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the NeMo STT provider.

        Args:
            config: Provider configuration with optional keys:
                - model_path: Path to ``nemotron_asr.pt``
                  (default: env ``NEMO_ASR_MODEL_PATH`` or ``weights/nemotron_asr.pt``).
                - device: Compute device "cuda" or "cpu"
                  (default: "cuda" if available, else "cpu").
        """
        self.model_path = config.get("model_path") or os.getenv(
            "NEMO_ASR_MODEL_PATH", "weights/nemotron_asr.pt"
        )
        device = config.get("device") or os.getenv("NEMO_DEVICE")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the NemotronASR checkpoint, falling back to CPU on GPU failure."""
        try:
            # Lazy import so the STT registry can be imported without eic_speech_nemo
            # installed (only the "nemo" provider needs it).
            from eic_speech_nemo.models.nemo import NemotronASR
        except ImportError as exc:
            raise ImportError(
                "eic-speech-nemo is required for the NeMo provider. "
                "Install the sibling repo as an editable dependency."
            ) from exc

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"NeMo ASR weights not found at '{self.model_path}'. "
                "Download nemotron_asr.pt from "
                "https://huggingface.co/chalkp/eic-speech-nemo and set NEMO_ASR_MODEL_PATH."
            )

        logger.info("Loading NeMo ASR: %s (device=%s)", self.model_path, self.device)
        try:
            self.model = NemotronASR.load_from_pt(self.model_path, device=self.device)
        except Exception as exc:
            if self.device != "cpu":
                logger.warning(
                    "Failed to load NeMo ASR on '%s' (%s); falling back to CPU...",
                    self.device,
                    exc,
                )
                self.device = "cpu"
                self.model = NemotronASR.load_from_pt(self.model_path, device="cpu")
            else:
                logger.error("Failed to load NeMo ASR model: %s", exc)
                raise
        self.model.eval()
        logger.info("NeMo ASR loaded (device=%s)", self.device)

    def transcribe(self, audio_content: bytes, **kwargs) -> str:
        """Transcribe audio to text.

        Args:
            audio_content: Raw PCM audio (16-bit signed, 16 kHz, mono). A leading
                44-byte WAV/RIFF header is stripped if present.
            **kwargs: Provider-specific options (unused).

        Returns:
            Transcribed text.
        """
        if self.model is None:
            raise RuntimeError("NeMo ASR model not loaded")

        logger.debug("Transcribing %d bytes of audio", len(audio_content))

        if audio_content[:4] == b"RIFF":
            audio_content = audio_content[_WAV_HEADER_LEN:]

        # 16-bit PCM bytes -> normalized float32 in [-1, 1] (same as the whisper provider).
        audio_array = np.frombuffer(audio_content, dtype=np.int16).astype(np.float32) / 32768.0
        if audio_array.size == 0:
            return ""

        # .copy() because np.frombuffer returns a read-only, non-owning array.
        tensor = torch.from_numpy(audio_array.copy()).to(self.device)
        with torch.no_grad():
            text = self.model.transcribe(tensor)

        text = (text or "").strip()
        logger.debug("Transcribed: '%s'", text)
        return text

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return self.SUPPORTED_LANGUAGES.copy()
