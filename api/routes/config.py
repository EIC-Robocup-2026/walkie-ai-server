"""Loader for ``api/routes/config.toml`` — the API layer's central tunables.

Secrets stay in ``.env`` (``ELEVENLABS_API_KEY``, ``GOOGLE_*`` credentials);
everything tweakable — provider selection, model paths/checkpoints, devices,
and the vision debug toggle — lives in ``config.toml`` so it can be changed
without editing code. The file is parsed once, at import time.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).with_name("config.toml")

with _CONFIG_PATH.open("rb") as _f:
    CONFIG: dict[str, Any] = tomllib.load(_f)


def section(name: str) -> dict[str, Any]:
    """Return the top-level ``[name]`` table (empty dict if missing)."""
    value = CONFIG.get(name)
    return dict(value) if isinstance(value, dict) else {}


def compact(mapping: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is ``None`` or ``""``.

    An empty string in config.toml means "unset — use the provider default",
    mirroring the old ``if env_var:`` guards around provider kwargs.
    """
    return {k: v for k, v in mapping.items() if v not in (None, "")}


# services/debug_viewer.py reads VISION_DEBUG from the environment to keep the
# services layer decoupled from the api layer. Mirror the toml value into the
# env at load time so config.toml stays the single source of truth.
os.environ["VISION_DEBUG"] = "true" if section("vision").get("debug") else "false"
