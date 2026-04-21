"""Image Captioning module with configurable providers.

Usage:
    from src.vision import ImageCaption
    
    # Initialize with default provider (google)
    captioner = ImageCaption()
    
    # Or with explicit config
    captioner = ImageCaption(provider="google", model="gemini-2.5-flash")
    
    # Caption an image
    text = captioner.caption(image_bytes)
    
    # Caption with a custom prompt
    text = captioner.caption(image_bytes, prompt="What objects are in this image?")
"""

from typing import Union

from PIL import Image

from .base import ImageCaptionProvider
from .providers import get_provider, list_providers


class ImageCaption:
    """Image Captioning interface with configurable providers."""

    def __init__(
        self,
        provider: str = "google",
        **provider_config,
    ) -> None:
        """Initialize ImageCaption with a provider.
        
        Args:
            provider: The name of the provider to use (e.g., "google").
            **provider_config: Provider-specific configuration passed directly.
        """
        self._provider_name = provider
        self._provider: ImageCaptionProvider = get_provider(provider, provider_config)

    @property
    def provider_name(self) -> str:
        """Get the current provider name."""
        return self._provider_name

    @property
    def provider(self) -> ImageCaptionProvider:
        """Get the underlying provider instance."""
        return self._provider

    def load_model(self) -> None:
        """Pre-load the provider's model weights into memory."""
        self._provider.load_model()

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
        return self._provider.caption(image, prompt)

    def caption_batch(
        self,
        images: list[Union[bytes, Image.Image]],
        prompts: list[str] | None = None,
    ) -> list[str]:
        """Generate captions for multiple images.
        
        Args:
            images: List of images to caption (bytes or PIL Image).
            prompts: Optional list of prompts; if None, uses the provider's default prompt.
                     If provided, must be the same length as images.
            
        Returns:
            List of caption strings, one per image, in the same order as images.
        """
        return self._provider.caption_batch(images, prompts)

    def get_supported_formats(self) -> list[str]:
        """Get list of supported image formats for current provider."""
        return self._provider.get_supported_formats()

    def get_default_prompt(self) -> str:
        """Get the default prompt used by current provider."""
        return self._provider.get_default_prompt()

    @staticmethod
    def available_providers() -> list[str]:
        """List all available image captioning providers."""
        return list_providers()
