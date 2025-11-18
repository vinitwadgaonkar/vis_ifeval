"""OpenAI DALL-E model implementation."""

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.models.base_model import ImageModel


class OpenAIModel(ImageModel):
    """OpenAI DALL-E model for image generation."""

    def __init__(
        self,
        name: str = "openai",
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
    ) -> None:
        """Initialize OpenAI model.

        Args:
            name: Model name.
            model: OpenAI model name ("dall-e-3" or "dall-e-2").
            size: Image size ("1024x1024", "1792x1024", "1024x1792" for DALL-E 3).
            quality: Image quality ("standard" or "hd" for DALL-E 3).
        """
        super().__init__(
            name=name,
            config={
                "model": model,
                "size": size,
                "quality": quality,
            },
        )
        self.model_name = model
        self.size = size
        self.quality = quality
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

    def generate(self, prompt: str, seed: Optional[int] = None) -> "Image.Image":
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the desired image.
            seed: Optional random seed (not supported by OpenAI API, ignored).

        Returns:
            Generated PIL Image.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI model requires openai package. Install with: pip install openai"
            ) from e

        client = OpenAI(api_key=self.api_key)

        if self.model_name == "dall-e-3":
            response = client.images.generate(
                model=self.model_name,
                prompt=prompt,
                size=self.size,
                quality=self.quality,
                n=1,
            )
        else:  # dall-e-2
            response = client.images.generate(
                model=self.model_name,
                prompt=prompt,
                size=self.size,
                n=1,
            )

        image_url = response.data[0].url

        # Download and convert to PIL Image
        import requests
        from io import BytesIO
        from PIL import Image

        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content))
        return img

