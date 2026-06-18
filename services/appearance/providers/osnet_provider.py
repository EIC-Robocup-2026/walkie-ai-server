"""OSNet appearance re-ID provider (vendored ``osnet_x1_0``, pure PyTorch).

OSNet is a person re-identification model: it embeds clothing and body shape
into a 512-d vector that is stable within a session (a guest's attire doesn't
change), making it a reliable secondary identity key when the face is not
visible. This hosts Chalk's (EIC) reference pipeline from
``eic_human/pipeline/appearance.py`` — OSNet x1.0, L2-normalized output —
with no fine-tuning or dataset required.

The architecture is vendored in ``osnet_model`` (no ``torchreid`` dependency —
its build is broken and unmaintained) and the ImageNet-pretrained weights are
pulled from the Hugging Face Hub on first load, matching what torchreid's
``FeatureExtractor(model_path="")`` downloads. Preprocessing mirrors
torchreid's extractor exactly (RGB → resize 256x128 → ImageNet normalize) so
embeddings are equivalent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..base import AppearanceProvider
from .osnet_model import osnet_x1_0

# Default Hugging Face mirror of KaiyangZhou's official osnet weights — the
# ImageNet-pretrained checkpoint torchreid auto-downloads for ``osnet_x1_0``.
_DEFAULT_HF_REPO = "kaiyangzhou/osnet"
_DEFAULT_HF_FILE = "osnet_x1_0_imagenet.pth"

# torchreid FeatureExtractor preprocessing constants (ImageNet stats).
_INPUT_SIZE = (256, 128)  # (height, width)
_PIXEL_MEAN = (0.485, 0.456, 0.406)
_PIXEL_STD = (0.229, 0.224, 0.225)


class OSNetProvider(AppearanceProvider):
    """Appearance embedding via a vendored OSNet x1.0 + HF-hosted weights."""

    DEFAULT_MODEL = "osnet_x1_0"
    DIM = 512

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the OSNet provider.

        Args:
            config: Optional keys:
                - model: model name, for provenance only (default
                    ``"osnet_x1_0"``).
                - model_path: Local path to a weights ``.pth``. If set and the
                    file exists, it is used instead of downloading.
                - hf_repo / hf_filename: Hugging Face repo + file to fetch the
                    pretrained weights from (default KaiyangZhou's mirror).
                - device: ``"cuda"`` / ``"cpu"``. Default: auto (CUDA when
                    available).
        """
        self._config = config
        self._model_name: str = config.get("model", self.DEFAULT_MODEL)
        self._model_path: str = config.get("model_path", "")
        self._hf_repo: str = config.get("hf_repo", _DEFAULT_HF_REPO)
        self._hf_filename: str = config.get("hf_filename", _DEFAULT_HF_FILE)
        self._device: str | None = config.get("device")
        self._model: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Pre-load the network (and pretrained weights) into memory."""
        self._ensure_loaded()

    def _resolve_weights(self) -> str:
        """Return a local path to the OSNet weights, downloading if needed."""
        import os

        if self._model_path and os.path.isfile(self._model_path):
            return self._model_path
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:  # pragma: no cover - declared dependency
            raise ImportError(
                "OSNet appearance provider needs huggingface_hub to fetch "
                "pretrained weights. Install it, or set 'model_path' to a "
                "local osnet_x1_0 .pth."
            ) from e
        try:
            return hf_hub_download(repo_id=self._hf_repo, filename=self._hf_filename)
        except Exception as e:
            raise RuntimeError(
                f"Could not download OSNet weights '{self._hf_filename}' from "
                f"'{self._hf_repo}'. Set 'model_path' to a local osnet_x1_0 "
                f".pth, or 'hf_repo'/'hf_filename' to a reachable mirror. "
                f"Error: {e}"
            ) from e

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = osnet_x1_0(num_classes=1000, loss="softmax")

        weights_path = self._resolve_weights()
        # weights_only=True restricts unpickling to plain tensors/state-dicts,
        # so a tampered checkpoint can't execute code on load.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(
            checkpoint, dict
        ) else checkpoint
        # Drop DataParallel "module." prefixes and any classifier-shaped keys
        # that don't match (the imagenet checkpoint carries a 1000-way head;
        # we only need the 512-d feature trunk).
        model_dict = model.state_dict()
        matched = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            if k in model_dict and model_dict[k].shape == v.shape:
                matched[k] = v
        if not matched:
            raise RuntimeError(
                f"OSNet weights at '{weights_path}' did not match the model "
                "(no overlapping keys) — wrong/corrupt checkpoint."
            )
        model_dict.update(matched)
        model.load_state_dict(model_dict)

        model.eval()
        model.to(device)
        self._device = device
        self._model = model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _preprocess(self, image: Image.Image) -> "Any":
        """RGB PIL → normalized (1, 3, 256, 128) float tensor."""
        import torch

        h, w = _INPUT_SIZE
        rgb = image.convert("RGB").resize((w, h), Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0          # HxWx3, [0,1]
        mean = np.asarray(_PIXEL_MEAN, dtype=np.float32)
        std = np.asarray(_PIXEL_STD, dtype=np.float32)
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
        return tensor.to(self._device)

    def embed(self, image: Image.Image) -> list[float]:
        """Embed one person crop into an L2-normalized appearance vector."""
        self._ensure_loaded()
        assert self._model is not None
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            feats = self._model(self._preprocess(image))           # (1, 512)
            emb = F.normalize(feats, p=2, dim=1)[0].cpu().numpy()
        return [float(x) for x in emb]

    def get_model_name(self) -> str:
        return self._model_name

    def get_embedding_dim(self) -> int:
        return self.DIM
