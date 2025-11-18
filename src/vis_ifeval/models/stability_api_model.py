"""Stability AI API model implementation."""

import os
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.models.base_model import ImageModel

logger = logging.getLogger(__name__)


class StabilityAPIModel(ImageModel):
    """Stability AI API model for image generation.

    Uses Stability AI's official API for SDXL, SD3, and other models.
    """

    def __init__(
        self,
        name: str = "stability-api",
        api_key: Optional[str] = None,
        engine_id: str = "stable-diffusion-xl-1024-v1-0",
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        cfg_scale: float = 7.0,
    ) -> None:
        """Initialize Stability AI API model.

        Args:
            name: Model name.
            api_key: Stability AI API key. If None, reads from STABILITY_API_KEY env var.
            engine_id: Engine ID (e.g., "stable-diffusion-xl-1024-v1-0", "stable-diffusion-3-medium").
            width: Image width.
            height: Image height.
            steps: Number of inference steps.
            cfg_scale: Guidance scale.
        """
        api_key = api_key or os.getenv("STABILITY_API_KEY")
        if not api_key:
            raise ValueError(
                "STABILITY_API_KEY environment variable not set. "
                "Get your API key from https://platform.stability.ai/account/keys"
            )

        super().__init__(
            name=name,
            config={
                "engine_id": engine_id,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale,
            },
        )
        self.api_key = api_key
        self.engine_id = engine_id
        self.width = width
        self.height = height
        self.steps = steps
        self.cfg_scale = cfg_scale
        self._load_client()

    def _load_client(self) -> None:
        """Initialize the API client."""
        try:
            import requests
            self.requests = requests
        except ImportError as e:
            raise ImportError(
                "Stability API model requires the 'requests' library. Install with: pip install requests"
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
        import base64

        url = f"https://api.stability.ai/v1/generation/{self.engine_id}/text-to-image"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": self.cfg_scale,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "samples": 1,
        }

        if seed is not None:
            payload["seed"] = seed

        try:
            response = self.requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()

            result = response.json()

            # Stability AI returns base64 encoded images
            if "artifacts" in result and len(result["artifacts"]) > 0:
                image_data = result["artifacts"][0].get("base64")
                if image_data:
                    img_data = base64.b64decode(image_data)
                    img = Image.open(BytesIO(img_data))
                    return img
                else:
                    raise ValueError("No image data in response")
            else:
                raise ValueError(f"Unexpected response format: {result.keys()}")

        except Exception as e:
            logger.error(f"Stability AI API error: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"API error details: {error_detail}")
                except:
                    pass
            raise RuntimeError(f"Failed to generate image with Stability AI: {e}") from e

