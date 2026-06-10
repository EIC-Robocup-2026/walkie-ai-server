"""Appearance (attire) re-ID embedding module with pluggable providers."""

from .appearance import Appearance
from .base import AppearanceProvider

__all__ = [
    "Appearance",
    "AppearanceProvider",
]
