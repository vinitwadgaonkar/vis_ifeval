"""Replicate API model implementation."""

import os
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.models.base_model import ImageModel

logger = logging.getLogger(__name__)


class ReplicateModel(ImageModel):
    """Replicate API model for image generation.

    Replicate hosts many models including Stable Diffusion variants,
    FLUX, SDXL, etc. You can use any model from their platform.
    """

    def __init__(
        self,
        name: str = "replicate",
        api_token: Optional[str] = None,
        model_id: str = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> None:
        """Initialize Replicate model.

        Args:
            name: Model name.
            api_token: Replicate API token. If None, reads from REPLICATE_API_TOKEN env var.
            model_id: Replicate model ID (e.g., "stability-ai/sdxl:...", "black-forest-labs/flux-dev").
            width: Image width.
            height: Image height.
            num_inference_steps: Number of inference steps.
            guidance_scale: Guidance scale.
        """
        api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            raise ValueError(
                "REPLICATE_API_TOKEN environment variable not set. "
                "Get your API token from https://replicate.com/account/api-tokens"
            )

        super().__init__(
            name=name,
            config={
                "model_id": model_id,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )
        self.api_token = api_token
        self.model_id = model_id
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self._load_client()

    def _load_client(self) -> None:
        """Initialize the Replicate client."""
        try:
            import replicate
            self.client = replicate.Client(api_token=self.api_token)
        except ImportError as e:
            raise ImportError(
                "Replicate model requires the 'replicate' library. "
                "Install with: pip install replicate"
            ) from e

    def generate(self, prompt: str, seed: Optional[int] = None) -> "Image.Image":
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the desired image.
            seed: Optional random seed for reproducibility.

        Returns:
            Generated PIL Image.
        """
        import requests
        from io import BytesIO
        from PIL import Image

        try:
            # Replicate API
            input_params = {
                "prompt": prompt,
                "width": self.width,
                "height": self.height,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
            }

            if seed is not None:
                input_params["seed"] = seed

            output = self.client.run(self.model_id, input=input_params)

            # Replicate returns a URL or list of URLs
            if isinstance(output, list):
                image_url = output[0]
            else:
                image_url = output

            # Download the image
            response = requests.get(image_url, timeout=120)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))

            return img

        except Exception as e:
            logger.error(f"Replicate API error: {e}")
            raise RuntimeError(f"Failed to generate image with Replicate: {e}") from e

