"""NovelAI API model implementation."""

import os
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.models.base_model import ImageModel

logger = logging.getLogger(__name__)


class NovelAIModel(ImageModel):
    """NovelAI API model for image generation."""

    def __init__(
        self,
        name: str = "novelai",
        api_key: Optional[str] = None,
        model: str = "nai-diffusion-3",
        width: int = 1024,
        height: int = 1024,
        steps: int = 28,
        scale: float = 7.0,
    ) -> None:
        """Initialize NovelAI model.

        Args:
            name: Model name.
            api_key: NovelAI API key. If None, reads from NOVELAI_API_KEY env var.
            model: Model to use (e.g., "nai-diffusion-3", "nai-diffusion-furry").
            width: Image width.
            height: Image height.
            steps: Number of inference steps.
            scale: Guidance scale.
        """
        api_key = api_key or os.getenv("NOVELAI_API_KEY")
        if not api_key:
            raise ValueError(
                "NOVELAI_API_KEY environment variable not set. "
                "Get your API key from https://novelai.net/"
            )

        super().__init__(
            name=name,
            config={
                "model": model,
                "width": width,
                "height": height,
                "steps": steps,
                "scale": scale,
            },
        )
        self.api_key = api_key
        self.model = model
        self.width = width
        self.height = height
        self.steps = steps
        self.scale = scale
        self._load_client()

    def _load_client(self) -> None:
        """Initialize the API client."""
        try:
            import requests
            self.requests = requests
        except ImportError as e:
            raise ImportError(
                "NovelAI model requires the 'requests' library. Install with: pip install requests"
            ) from e

    def generate(self, prompt: str, seed: Optional[int] = None) -> "Image.Image":
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the desired image.
            seed: Optional random seed for reproducibility.

        Returns:
            Generated PIL Image.
        """
        from io import BytesIO
        from PIL import Image

        url = "https://api.novelai.net/ai/generate-image"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "input": prompt,
            "model": self.model,
            "parameters": {
                "width": self.width,
                "height": self.height,
                "scale": self.scale,
                "sampler": "k_euler_ancestral",
                "steps": self.steps,
                "n_samples": 1,
                "ucPreset": 0,
                "qualityToggle": True,
            },
        }

        if seed is not None:
            payload["parameters"]["seed"] = seed

        try:
            response = self.requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()

            # NovelAI returns image as base64 or binary
            if response.headers.get("content-type", "").startswith("image/"):
                img = Image.open(BytesIO(response.content))
            else:
                # Try to parse as JSON with base64 image
                data = response.json()
                import base64
                if "image" in data:
                    img_data = base64.b64decode(data["image"])
                    img = Image.open(BytesIO(img_data))
                else:
                    raise ValueError(f"Unexpected response format: {data.keys()}")

            return img

        except Exception as e:
            logger.error(f"NovelAI API error: {e}")
            raise RuntimeError(f"Failed to generate image with NovelAI: {e}") from e

