"""Appearance provider registry."""

from typing import Any

from ..base import AppearanceProvider
from .osnet_provider import OSNetProvider

PROVIDERS: dict[str, type[AppearanceProvider]] = {
    "osnet": OSNetProvider,
}


def get_provider(name: str, config: dict[str, Any]) -> AppearanceProvider:
    """Get an appearance provider instance by name."""
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown appearance provider: '{name}'. Available: {available}"
        )
    return provider_class(config)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDERS.keys())
