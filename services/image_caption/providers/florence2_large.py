"""Florence-2-large image captioning provider (local inference via HuggingFace Transformers)."""

import io
from typing import Any, Union

import torch
from PIL import Image
from transformers import Florence2ForConditionalGeneration, Florence2Processor

from ..base import ImageCaptionProvider


class Florence2LargeImageCaptionProvider(ImageCaptionProvider):
    """Florence-2-large image captioning provider using HuggingFace Transformers (local)."""

    DEFAULT_MODEL = "florence-community/Florence-2-large"
    DEFAULT_PROMPT = "<DETAILED_CAPTION>"
    DEFAULT_MAX_NEW_TOKENS = 1024
    DEFAULT_NUM_BEAMS = 3

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Florence-2-large image captioning provider.

        Args:
            config: Provider configuration with optional keys:
                - model: HuggingFace model id (default: "microsoft/Florence-2-large")
                - device: "cuda" or "cpu" (default: "cuda" if available)
                - default_prompt: Florence-2 task prompt (default: "<DETAILED_CAPTION>")
                - max_new_tokens: Max tokens to generate (default: 1024)
                - num_beams: Beam search width (default: 3)
        """
        self.model_name = config.get("model", self.DEFAULT_MODEL)
        device_str = config.get("device", "cuda")
        self.device = torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        self.default_prompt = config.get("default_prompt", self.DEFAULT_PROMPT)
        self.max_new_tokens = config.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS)
        self.num_beams = config.get("num_beams", self.DEFAULT_NUM_BEAMS)

        self.processor: Florence2Processor | None = None
        self.model: Florence2ForConditionalGeneration | None = None

    def load_model(self) -> None:
        """Pre-load Florence-2-large model and processor into memory."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Lazy-load Florence-2-large model and processor on first use."""
        if self.model is not None:
            return
        self.processor = Florence2Processor.from_pretrained(
            self.model_name
        )
        self.model = (
            Florence2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
            )
            .eval()
            .to(self.device)
        )

    def _to_pil(self, image: Union[bytes, Image.Image]) -> Image.Image:
        """Convert image to RGB PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(io.BytesIO(image)).convert("RGB")

    def _run_inference(self, pil_image: Image.Image, task_prompt: str) -> str:
        """Run Florence-2 inference and return the parsed caption string.

        Args:
            pil_image: RGB PIL Image.
            task_prompt: Florence-2 task token (e.g. "<DETAILED_CAPTION>").

        Returns:
            Parsed caption string.
        """
        assert self.processor is not None and self.model is not None

        inputs = self.processor(
            text=task_prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                early_stopping=False,
                do_sample=False,
                num_beams=self.num_beams,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(pil_image.width, pil_image.height),
        )
        return parsed.get(task_prompt, generated_text).strip()

    def caption(
        self,
        image: Union[bytes, Image.Image],
        prompt: str | None = None,
    ) -> str:
        """Generate a caption or description for an image.

        Args:
            image: The image to caption, either as bytes or PIL Image.
            prompt: Optional Florence-2 task prompt (e.g. "<DETAILED_CAPTION>").
                    If None, uses default_prompt.

        Returns:
            The generated caption/description as a string.
        """
        self._ensure_loaded()
        task_prompt = prompt if prompt is not None else self.default_prompt
        pil_image = self._to_pil(image)
        return self._run_inference(pil_image, task_prompt)

    def caption_batch(
        self,
        images: list[Union[bytes, Image.Image]],
        prompts: list[str] | None = None,
    ) -> list[str]:
        """Generate captions for multiple images.

        Florence-2 does not natively support batched image+text pairs in
        the same way as encoder-only VLMs, so this iterates sequentially.

        Args:
            images: List of images to caption (bytes or PIL Image).
            prompts: Optional list of task prompts; if None, uses default_prompt.
                     If provided, must be the same length as images.

        Returns:
            List of caption strings, one per image, in the same order as images.
        """
        if not images:
            return []

        self._ensure_loaded()

        if prompts is None:
            prompts = [self.default_prompt] * len(images)
        elif len(prompts) != len(images):
            raise ValueError("Number of prompts must match number of images")

        return [
            self._run_inference(self._to_pil(img), p)
            for img, p in zip(images, prompts)
        ]

    def get_supported_formats(self) -> list[str]:
        """Get list of supported image formats."""
        return ["jpeg", "png"]

    def get_default_prompt(self) -> str:
        """Get the default prompt used when none is provided."""
        return self.default_prompt
