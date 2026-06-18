"""Unit tests for the NeMo STT provider (no running server required).

The transcription test needs the nemotron_asr.pt checkpoint; it is skipped when
the weights are absent so the suite still passes on machines without them.
"""

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _weights_path() -> str:
    return os.getenv("NEMO_ASR_MODEL_PATH") or str(_ROOT / "weights" / "nemotron_asr.pt")


def test_nemo_is_registered():
    """The provider is importable and registered (no weights / eic_speech_nemo needed)."""
    from services.stt.providers import PROVIDERS, list_providers

    assert "nemo" in PROVIDERS
    assert "nemo" in list_providers()


@pytest.mark.skipif(
    not os.path.exists(_weights_path()),
    reason="nemotron_asr.pt weights not present",
)
def test_nemo_transcribe_returns_string(sample_audio_bytes):
    from services.stt.providers.nemo import NemoSTTProvider

    provider = NemoSTTProvider({"model_path": _weights_path()})
    text = provider.transcribe(sample_audio_bytes)  # silent WAV -> (likely empty) string
    assert isinstance(text, str)
    assert isinstance(provider.get_supported_languages(), list)
