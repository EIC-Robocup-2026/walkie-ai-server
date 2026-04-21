"""CLIP embedding provider (OpenAI CLIP ViT-B/16)."""

from typing import Any

import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Ensure HF_TOKEN from .env is available for Hugging Face downloads and API
load_dotenv()

from ..base import EmbeddingProvider


class CLIPEmbeddingProvider(EmbeddingProvider):
    """CLIP image and text embedding provider using transformers."""

    DEFAULT_MODEL = "openai/clip-vit-base-patch16"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize CLIP provider.

        Args:
            config: Optional keys:
                - model: HuggingFace model id (default: openai/clip-vit-base-patch16)
                - device: "cuda" or "cpu" (default: auto)
        """
        self.model_id = config.get("model", self.DEFAULT_MODEL)
        device = config.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self._dim: int | None = None

    def load_model(self) -> None:
        """Pre-load CLIP model and processor into memory."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Lazy-load model and processor on first use."""
        if self._model is not None:
            return
        self._model = CLIPModel.from_pretrained(self.model_id).to(self._device)
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        # CLIP ViT-B/16 has 512-dim embeddings
        self._dim = self._model.config.projection_dim

    def _to_tensor(self, out: Any) -> torch.Tensor:
        """Extract embedding tensor from model output (handles BaseModelOutputWithPooling)."""
        if isinstance(out, torch.Tensor):
            return out
        # Newer transformers may return an output object; avoid "tensor or x" (boolean of tensor is ambiguous)
        t = getattr(out, "pooler_output", None)
        if t is None:
            t = getattr(out, "last_hidden_state", None)
        if t is not None:
            if t.dim() > 2:
                t = t[:, 0]
            return t
        raise TypeError(f"Unexpected model output type: {type(out)}")

    def embed_image(self, image: Image.Image) -> list[float]:
        """Compute normalized CLIP image embedding."""
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.get_image_features(**inputs)
            features = self._to_tensor(out)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()

    def embed_text(self, text: str) -> list[float]:
        """Compute normalized CLIP text embedding."""
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None
        inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
            features = self._to_tensor(out)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch text embeddings for efficiency."""
        if not texts:
            return []
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None
        inputs = self._processor(text=texts, return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
            features = self._to_tensor(out)
            features = features / features.norm(dim=-1, keepdim=True)
        return [row.tolist() for row in features.cpu().numpy()]

    def get_embedding_dim(self) -> int:
        """Return CLIP projection dimension (512 for ViT-B/16)."""
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim
