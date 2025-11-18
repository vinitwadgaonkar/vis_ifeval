"""Banana.dev API model implementation."""

import os
import logging
from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.models.base_model import ImageModel

logger = logging.getLogger(__name__)


class BananaModel(ImageModel):
    """Banana.dev API model for image generation.

    Banana.dev is a serverless GPU platform. You can deploy any model there
    and call it via API. This wrapper works with Stable Diffusion models
    deployed on Banana.dev.
    """

    def __init__(
        self,
        name: str = "banana",
        api_key: Optional[str] = None,
        model_key: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> None:
        """Initialize Banana.dev model.

        Args:
            name: Model name.
            api_key: Banana.dev API key. If None, reads from BANANA_API_KEY env var.
            model_key: Banana.dev model key. If None, reads from BANANA_MODEL_KEY env var.
            width: Image width.
            height: Image height.
            num_inference_steps: Number of inference steps.
            guidance_scale: Guidance scale.
        """
        api_key = api_key or os.getenv("BANANA_API_KEY")
        model_key = model_key or os.getenv("BANANA_MODEL_KEY")

        if not api_key:
            raise ValueError(
                "BANANA_API_KEY environment variable not set. "
                "Get your API key from https://banana.dev/"
            )
        if not model_key:
            raise ValueError(
                "BANANA_MODEL_KEY environment variable not set. "
                "This is the model deployment key from your Banana.dev dashboard."
            )

        super().__init__(
            name=name,
            config={
                "model_key": model_key,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )
        self.api_key = api_key
        self.model_key = model_key
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self._load_client()

    def _load_client(self) -> None:
        """Initialize the API client."""
        try:
            import requests
            self.requests = requests
        except ImportError as e:
            raise ImportError(
                "Banana model requires the 'requests' library. Install with: pip install requests"
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

        url = f"https://api.banana.dev/start/v4"
        headers = {
            "Content-Type": "application/json",
        }

        # Banana.dev expects this payload structure
        payload = {
            "apiKey": self.api_key,
            "modelKey": self.model_key,
            "modelInputs": {
                "prompt": prompt,
                "width": self.width,
                "height": self.height,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
            },
        }

        if seed is not None:
            payload["modelInputs"]["seed"] = seed

        try:
            # Start the job
            response = self.requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()

            if "id" not in result:
                raise ValueError(f"Unexpected response: {result}")

            call_id = result["id"]

            # Poll for results (Banana.dev is async)
            import time
            max_attempts = 60
            for attempt in range(max_attempts):
                time.sleep(2)  # Wait 2 seconds between polls

                check_url = f"https://api.banana.dev/check/v4"
                check_payload = {
                    "apiKey": self.api_key,
                    "id": call_id,
                }

                check_response = self.requests.post(
                    check_url, json=check_payload, headers=headers, timeout=30
                )
                check_response.raise_for_status()
                check_result = check_response.json()

                if check_result.get("finished"):
                    model_outputs = check_result.get("modelOutputs", [])
                    if model_outputs and len(model_outputs) > 0:
                        output = model_outputs[0]

                        # Banana.dev typically returns base64 encoded images
                        if "image" in output:
                            img_data = base64.b64decode(output["image"])
                            img = Image.open(BytesIO(img_data))
                            return img
                        elif "image_base64" in output:
                            img_data = base64.b64decode(output["image_base64"])
                            img = Image.open(BytesIO(img_data))
                            return img
                        else:
                            raise ValueError(f"Unexpected output format: {output.keys()}")

                if check_result.get("failed"):
                    error_msg = check_result.get("error", "Unknown error")
                    raise RuntimeError(f"Banana.dev job failed: {error_msg}")

            raise TimeoutError("Banana.dev job timed out after 2 minutes")

        except Exception as e:
            logger.error(f"Banana.dev API error: {e}")
            raise RuntimeError(f"Failed to generate image with Banana.dev: {e}") from e

