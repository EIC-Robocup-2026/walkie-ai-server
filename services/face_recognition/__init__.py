"""Face recognition module with pluggable providers."""

from .base import FaceEmbedding, FaceRecognitionProvider
from .face_recognition import FaceRecognition

__all__ = [
    "FaceEmbedding",
    "FaceRecognition",
    "FaceRecognitionProvider",
]
