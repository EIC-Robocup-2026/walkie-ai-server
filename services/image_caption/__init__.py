"""Image Captioning module with pluggable providers."""

from .base import ImageCaptionProvider
from .image_caption import ImageCaption

__all__ = ["ImageCaption", "ImageCaptionProvider"]
