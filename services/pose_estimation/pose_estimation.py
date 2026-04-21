"""Pose estimation module with configurable providers.

Usage:
    from src.vision.pose_estimation import PoseEstimation

    pose = PoseEstimation(provider="yolo_pose")
    persons = pose.estimate(pil_image)
"""

from typing import Any

from PIL import Image

from .base import PersonPose, PoseEstimationProvider
from .providers import get_provider, list_providers


class PoseEstimation:
    """Pose estimation interface with configurable providers."""

    def __init__(self, provider: str = "yolo_pose", **provider_config: Any) -> None:
        """Initialize PoseEstimation with a provider.

        Args:
            provider: Provider name (e.g., ``"yolo_pose"``).
            **provider_config: Provider-specific configuration.
        """
        self._provider_name = provider
        self._provider: PoseEstimationProvider = get_provider(
            provider, provider_config
        )

    @property
    def provider_name(self) -> str:
        """Current provider name."""
        return self._provider_name

    @property
    def provider(self) -> PoseEstimationProvider:
        """Underlying provider instance."""
        return self._provider

    def load_model(self) -> None:
        """Pre-load the provider's model weights into memory."""
        self._provider.load_model()

    def estimate(self, image: Image.Image) -> list[PersonPose]:
        """Estimate poses for all persons in *image*."""
        return self._provider.estimate(image)

    def get_model_name(self) -> str:
        """Model name for logging."""
        return self._provider.get_model_name()

    @staticmethod
    def available_providers() -> list[str]:
        """List all available pose estimation providers."""
        return list_providers()
