"""ElevenLabs TTS provider implementation."""

import os
from typing import Any, Iterator

from elevenlabs.client import ElevenLabs

from ..base import TTSProvider

def _build_style_prompt(personality: str | None = None) -> str:
    """Build the style prompt with optional personality customization."""
    personality_line = f"\nPersonality: {personality}. Choose tags matching this personality." if personality else ""
    
    return f"""Add audio tags to make text expressive for speech synthesis. Preserve original words exactly.

Voice tags (preferred - control vocal delivery and emotion):
- Laughter: [laughs], [laughs harder], [starts laughing], [chuckles], [wheezing], [snorts]
- Breath: [whispers], [sighs], [exhales]
- Emotion: [sarcastic], [curious], [excited], [crying], [mischievously], [happy], [sad], [angry], [annoyed], [thoughtful], [surprised]
{personality_line}
Examples:
- "Are you serious? I can't believe it!" → "[sighs] Are you serious? [laughs] I can't believe it!"
- "Wait, what did you just say?" → "[curious] Wait, what did you just say?"
- "That's the funniest thing I've heard" → "[starts laughing] That's the funniest thing I've heard! [wheezing]"

Reply with ONLY the enhanced text, no explanations."""

class ElevenLabsProvider(TTSProvider):
    """ElevenLabs Text-to-Speech provider using official SDK."""

    SUPPORTED_FORMATS = [
        "mp3_44100_128",
        "mp3_22050_32",
        "pcm_16000",
        "pcm_22050",
        "pcm_24000",
        "pcm_44100",
    ]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize ElevenLabs provider.
        
        Args:
            config: Provider configuration with optional keys:
                - voice_id: Voice ID (default: "21m00Tcm4TlvDq8ikWAM" - Rachel)
                - model_id: Model ID (default: "eleven_multilingual_v2")
                - output_format: Output format (default: "pcm_24000")
                - stability: Voice stability 0-1 (default: 0.5)
                - similarity_boost: Similarity boost 0-1 (default: 0.75)
                - style: Style 0-1 (default: 0.0)
                - use_speaker_boost: Enable speaker boost (default: True)
                - language_code: Language code for synthesis (optional)
        """
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")

        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        self.model_id = config.get("model_id", "eleven_multilingual_v2")
        self.output_format = config.get("output_format", "pcm_24000")
        self.language_code = config.get("language_code")

        self.voice_settings = {
            "stability": config.get("stability", 0.5),
            "similarity_boost": config.get("similarity_boost", 0.75),
            "style": config.get("style", 0.0),
            "use_speaker_boost": config.get("use_speaker_boost", True),
        }

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio."""
        audio_stream = self.synthesize_stream(text)
        return b"".join(audio_stream)

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Synthesize text to audio with streaming."""
        return self.client.text_to_speech.stream(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model_id,
            output_format=self.output_format,
            voice_settings=self.voice_settings,
            language_code=self.language_code,
        )

    def get_supported_formats(self) -> list[str]:
        """Get list of supported output formats."""
        return self.SUPPORTED_FORMATS.copy()

    def supports_streaming(self) -> bool:
        """ElevenLabs supports streaming."""
        return True
    
    @staticmethod
    def style_text(model, text: str, personality: str | None = None) -> str:
        """Enhance text with ElevenLabs v3 audio tags for expressive TTS output.
        
        Uses an LLM to intelligently add audio tags like [laughing], [sighs], 
        [excited], etc. to make the speech more natural and expressive.
        
        Args:
            model: The LLM model instance to use for enhancement (e.g., ChatOpenAI)
            text: The original text to enhance
            personality: Optional personality description to influence tag selection.
                Examples: "cheerful and energetic", "calm and professional",
                "playful and mischievous", "serious and thoughtful"
                
        Returns:
            The text enhanced with audio tags
        """
        prompt = _build_style_prompt(personality)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
        response = model.invoke(messages)
        return response.content