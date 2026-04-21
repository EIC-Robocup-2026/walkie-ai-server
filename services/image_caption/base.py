"""Base class for Image Captioning providers."""

from abc import ABC, abstractmethod
from typing import Union

from PIL import Image


class ImageCaptionProvider(ABC):
    """Abstract base class for image captioning providers.
    
    Implement this class to add a new image captioning provider.
    """

    @abstractmethod
    def caption(
        self,
        image: Union[bytes, Image.Image],
        prompt: str | None = None,
    ) -> str:
        """Generate a caption or description for an image.
        
        Args:
            image: The image to caption, either as bytes or PIL Image.
            prompt: Optional prompt to guide the captioning (e.g., "What is in this image?").
                   If None, uses the provider's default prompt.
            
        Returns:
            The generated caption/description as a string.
        """
        pass

    def caption_batch(
        self,
        images: list[Union[bytes, Image.Image]],
        prompts: list[str] | None = None,
    ) -> list[str]:
        """Generate captions for multiple images.
        
        Default implementation calls caption() for each image. Override for
        provider-specific batch processing (e.g. batched inference).
        
        Args:
            images: List of images to caption (bytes or PIL Image).
            prompts: Optional list of prompts; if None, uses the provider's default prompt.
                     If provided, must be the same length as images.
            
        Returns:
            List of caption strings, one per image, in the same order as images.
        """
        return [self.caption(img, prompts) for img in images]

    def load_model(self) -> None:
        """Pre-load model weights into memory.

        Default implementation is a no-op. Override in providers that use
        lazy loading so that ``load_model()`` can be called eagerly.
        """

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Get list of supported image formats.
        
        Returns:
            List of format identifiers (e.g., ['jpeg', 'png', 'webp', 'gif']).
        """
        pass

    def get_default_prompt(self) -> str:
        """Get the default prompt used when none is provided.
        
        Returns:
            Default prompt string.
        """
        return "Describe this image in detail."
