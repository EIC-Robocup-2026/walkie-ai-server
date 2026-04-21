"""Object detection module with pluggable providers."""

from .base import DetectedObject, ObjectDetectionProvider
from .object_detection import ObjectDetection

__all__ = ["DetectedObject", "ObjectDetection", "ObjectDetectionProvider"]
