"""Grasp-pose generation service with configurable providers.

Usage:
    from services.grasp import Grasp

    g = Grasp(provider="graspnet")
    g.load_model()
    grasps = g.infer(cloud_xyz, score_threshold=0.0, max_grasps=20, antipodal=True)
    # -> [{"translation": [...], "rotation": [[...]], "width": ..., "score": ...}, ...]
"""

from typing import Any

import numpy as np

from .base import GraspProvider
from .providers import get_provider, list_providers


class Grasp:
    """Grasp-pose generation interface with configurable providers."""

    def __init__(self, provider: str = "graspnet", **provider_config: Any) -> None:
        """Initialize Grasp with a provider.

        Args:
            provider: Provider name (e.g., ``"graspnet"``).
            **provider_config: Provider-specific configuration.
        """
        self._provider_name = provider
        self._provider: GraspProvider = get_provider(provider, provider_config)

    @property
    def provider_name(self) -> str:
        """Current provider name."""
        return self._provider_name

    @property
    def provider(self) -> GraspProvider:
        """Underlying provider instance."""
        return self._provider

    @property
    def rotation_offset(self) -> np.ndarray:
        """The 3x3 roll/pitch/yaw offset the provider applies to each output rotation."""
        return getattr(self._provider, "rotation_offset", np.eye(3))

    def load_model(self) -> None:
        """Pre-load the provider's model weights into memory."""
        self._provider.load_model()

    def infer(self, cloud: np.ndarray, **opts: Any) -> list[dict]:
        """Generate grasp poses for an ``(N, 3)`` cloud in its own frame."""
        return self._provider.infer(cloud, **opts)

    def get_model_name(self) -> str:
        """Model name for logging / provenance."""
        return self._provider.get_model_name()

    @staticmethod
    def available_providers() -> list[str]:
        """List all available grasp providers."""
        return list_providers()
