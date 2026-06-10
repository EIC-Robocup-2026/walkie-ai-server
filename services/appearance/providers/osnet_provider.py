"""OSNet appearance re-ID provider (torchreid, ``osnet_x1_0``).

OSNet is a person re-identification model: it embeds clothing and body shape
into a 512-d vector that is stable within a session (a guest's attire doesn't
change), making it a reliable secondary identity key when the face is not
visible. This hosts Chalk's (EIC) reference pipeline from
``eic_human/pipeline/appearance.py`` — OSNet x1.0 via ``torchreid``,
L2-normalized output — with no fine-tuning or dataset required.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..base import AppearanceProvider

# ---------------------------------------------------------------------------
# Lazy import guard – avoid pulling in torchreid at module level (matches the
# InsightFace/YOLO providers, and keeps a missing/broken model from blocking
# server startup — the route lazy-loads on first request).
# ---------------------------------------------------------------------------
_torchreid_imported = False


def _ensure_torchreid() -> None:
    global _torchreid_imported
    if _torchreid_imported:
        return
    try:
        import torchreid  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "OSNet appearance provider requires torchreid (deep-person-reid). "
            "It has a PEP 517 build-isolation bug, so install numpy+cython "
            "first and pass --no-build-isolation: "
            "pip install numpy cython && pip install --no-build-isolation "
            "git+https://github.com/KaiyangZhou/deep-person-reid.git"
        ) from e
    _torchreid_imported = True


class OSNetProvider(AppearanceProvider):
    """Appearance embedding via torchreid ``FeatureExtractor``."""

    DEFAULT_MODEL = "osnet_x1_0"
    DIM = 512

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the OSNet provider.

        Args:
            config: Optional keys:
                - model: torchreid model name (default ``"osnet_x1_0"``).
                - model_path: Path to weights; ``""`` auto-downloads the
                    pretrained checkpoint (default ``""``).
                - device: ``"cuda"`` / ``"cpu"``. Default: auto (CUDA when
                    available).
        """
        self._config = config
        self._model_name: str = config.get("model", self.DEFAULT_MODEL)
        self._model_path: str = config.get("model_path", "")
        self._device: str | None = config.get("device")
        self._extractor: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Pre-load the extractor (and pretrained weights) into memory."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._extractor is not None:
            return
        _ensure_torchreid()
        import torch
        from torchreid.utils import FeatureExtractor

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._extractor = FeatureExtractor(
            model_name=self._model_name,
            model_path=self._model_path,
            device=device,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def embed(self, image: Image.Image) -> list[float]:
        """Embed one person crop into an L2-normalized appearance vector."""
        self._ensure_loaded()
        assert self._extractor is not None
        import torch
        import torch.nn.functional as F

        # torchreid expects RGB HxWx3 uint8 arrays and handles its own
        # resize (256x128) + ImageNet normalization.
        rgb = np.asarray(image.convert("RGB"))

        with torch.no_grad():
            feats = self._extractor([rgb])                      # (1, 512)
            emb = F.normalize(feats, p=2, dim=1)[0].cpu().numpy()
        return [float(x) for x in emb]

    def get_model_name(self) -> str:
        return self._model_name

    def get_embedding_dim(self) -> int:
        return self.DIM
