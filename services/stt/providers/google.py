"""Google Cloud Speech-to-Text provider implementation."""

import os
from typing import Any

from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from ..base import STTProvider


class GoogleSTTProvider(STTProvider):
    """Google Cloud Speech-to-Text provider using official SDK."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Google STT provider.
        
        Args:
            config: Provider configuration with optional keys:
                - location: GCP region (default: "us-central1")
                - language_codes: List of language codes (default: ["en-US"])
                - model: Model name (default: "chirp_2")
                - recognizer: Recognizer name (default: "_")
        """
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")

        self.location = config.get("location", "us-central1")
        self.language_codes = config.get("language_codes", ["en-US"])
        self.model = config.get("model", "chirp_2")
        
        # Initialize client
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        client_options = ClientOptions(
            api_endpoint=f"{self.location}-speech.googleapis.com"
        )
        
        if credentials_path:
            self.client = SpeechClient.from_service_account_file(
                credentials_path, client_options=client_options
            )
        else:
            self.client = SpeechClient(client_options=client_options)

        # Build recognizer path
        recognizer = config.get("recognizer", "_")
        self.recognizer_path = (
            f"projects/{self.project_id}/locations/{self.location}/recognizers/{recognizer}"
        )

    def transcribe(self, audio_content: bytes, **kwargs) -> str:
        """Transcribe audio to text.
        
        Args:
            audio_content: Raw PCM audio (16-bit signed, 16kHz, mono).
            **kwargs: Provider-specific options (unused).
        """
        # Explicit config for raw PCM audio from microphone
        config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                audio_channel_count=1,
            ),
            language_codes=self.language_codes,
            model=self.model,
        )

        request = cloud_speech.RecognizeRequest(
            recognizer=self.recognizer_path,
            config=config,
            content=audio_content,
        )

        response = self.client.recognize(request=request)

        transcripts = []
        for result in response.results:
            if result.alternatives:
                transcripts.append(result.alternatives[0].transcript)

        return " ".join(transcripts)

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return [
            "en-US", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR",
            "ja-JP", "ko-KR", "zh-CN", "th-TH", "vi-VN", "id-ID",
        ]
