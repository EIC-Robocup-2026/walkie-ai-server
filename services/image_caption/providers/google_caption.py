"""Google Generative AI image captioning provider implementation."""

import base64
import io
from typing import Any, Union

from PIL import Image

from google import genai
from google.genai.types import HttpOptions, Part

from ..base import ImageCaptionProvider


class GoogleImageCaptionProvider(ImageCaptionProvider):
    """Google Generative AI image captioning provider using google-genai SDK."""

    DEFAULT_MODEL = "gemini-2.5-flash"
    DEFAULT_PROMPT = "Describe this image in detail."

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Google image captioning provider.
        
        Args:
            config: Provider configuration with optional keys:
                - model: Model name (default: "gemini-2.5-flash")
                - api_version: API version (default: "v1")
                - default_prompt: Default prompt when none provided
        
        Environment variables:
            - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON (optional)
            - GOOGLE_API_KEY: API key for authentication (optional, alternative to ADC)
        """
        self.model = config.get("model", self.DEFAULT_MODEL)
        self.default_prompt = config.get("default_prompt", self.DEFAULT_PROMPT)
        api_version = config.get("api_version", "v1")
        
        # Initialize client
        # google-genai SDK automatically handles credentials:
        # 1. GOOGLE_API_KEY environment variable
        # 2. GOOGLE_APPLICATION_CREDENTIALS for service account
        # 3. Application Default Credentials (ADC)
        self.client = genai.Client(http_options=HttpOptions(api_version=api_version))

    def load_model(self) -> None:
        """No-op for Google provider (API-based, no local model to load)."""

    def _image_to_part(self, image: Union[bytes, Image.Image]) -> Part:
        """Convert image to google-genai Part.
        
        Args:
            image: Image as bytes or PIL Image.
            
        Returns:
            Part object for the API request.
        """
        # Convert PIL Image to bytes if needed
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            # Determine format based on image mode
            img_format = "PNG" if image.mode == "RGBA" else "JPEG"
            image.save(buffer, format=img_format)
            image_bytes = buffer.getvalue()
            mime_type = "image/png" if img_format == "PNG" else "image/jpeg"
        else:
            image_bytes = image
            # Try to detect mime type from bytes
            mime_type = self._detect_mime_type(image_bytes)
        
        # Encode as base64 for inline data
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        return Part.from_bytes(data=image_bytes, mime_type=mime_type)

    def _detect_mime_type(self, image_bytes: bytes) -> str:
        """Detect MIME type from image bytes.
        
        Args:
            image_bytes: Raw image bytes.
            
        Returns:
            MIME type string.
        """
        # Check magic bytes
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            return "image/jpeg"
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            return "image/gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            return "image/webp"
        else:
            # Default to JPEG
            return "image/jpeg"

    def caption(
        self,
        image: Union[bytes, Image.Image],
        prompt: str | None = None,
    ) -> str:
        """Generate a caption or description for an image.
        
        Args:
            image: The image to caption, either as bytes or PIL Image.
            prompt: Optional prompt to guide the captioning.
            
        Returns:
            The generated caption/description as a string.
        """
        if prompt is None:
            prompt = self.default_prompt
        
        image_part = self._image_to_part(image)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image_part],
        )
        
        return response.text

    def caption_batch(
        self,
        images: list[Union[bytes, Image.Image]],
        prompts: list[str] | None = None,
    ) -> list[str]:
        """Generate captions for multiple images via sequential API calls.
        
        Args:
            images: List of images to caption (bytes or PIL Image).
            prompts: Optional list of prompts; if None, uses default_prompt.
                     If provided, must be the same length as images.
            
        Returns:
            List of caption strings, one per image, in the same order as images.
        """
        if prompts is None:
            prompts = [self.default_prompt] * len(images)
        if len(prompts) != len(images):
            raise ValueError("Number of prompts must match number of images")
        return [self.caption(img, prompt) for img, prompt in zip(images, prompts)]

    def get_supported_formats(self) -> list[str]:
        """Get list of supported image formats.
        
        Returns:
            List of format identifiers.
        """
        return ["jpeg", "png", "gif", "webp"]

    def get_default_prompt(self) -> str:
        """Get the default prompt used when none is provided.
        
        Returns:
            Default prompt string.
        """
        return self.default_prompt
