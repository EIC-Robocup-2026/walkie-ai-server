"""Grasp provider registry."""

from typing import Any

from ..base import GraspProvider
from .graspnet_provider import GraspNetProvider

PROVIDERS: dict[str, type[GraspProvider]] = {
    "graspnet": GraspNetProvider,
}


def get_provider(name: str, config: dict[str, Any]) -> GraspProvider:
    """Get a grasp provider instance by name."""
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown grasp provider: '{name}'. Available: {available}"
        )
    return provider_class(config)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDERS.keys())
