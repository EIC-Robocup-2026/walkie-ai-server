"""Pose estimation provider registry."""

from typing import Any

from ..base import PoseEstimationProvider
from .yolo_pose import YOLOPoseEstimationProvider

PROVIDERS: dict[str, type[PoseEstimationProvider]] = {
    "yolo_pose": YOLOPoseEstimationProvider,
}


def get_provider(name: str, config: dict[str, Any]) -> PoseEstimationProvider:
    """Get a pose estimation provider instance by name."""
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown pose estimation provider: '{name}'. Available: {available}"
        )
    return provider_class(config)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDERS.keys())
