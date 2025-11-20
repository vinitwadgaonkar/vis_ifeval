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
        model: str = "gpt-image-1",
        size: str = "1024x1024",
        quality: str = "high",
        organization: str | None = None,
    ) -> None:
        """Initialize OpenAI model.

        Args:
            name: Model name.
            model: OpenAI model name ("gpt-image-1", "dall-e-3", or "dall-e-2").
            size: Image size ("1024x1024", "1792x1024", "1024x1792").
            quality: Image quality. For gpt-image-1: "low", "medium", "high". 
                     For dall-e-3: "standard" or "hd".
            organization: Optional OpenAI organization ID.
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
        self.organization = organization or os.getenv("OPENAI_ORG_ID")

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

        # Initialize client with organization if provided
        client_kwargs = {"api_key": self.api_key}
        if self.organization:
            client_kwargs["organization"] = self.organization
        client = OpenAI(**client_kwargs)

        if self.model_name == "gpt-image-1":
            # GPT Image 1 model - uses quality parameter
            # Note: GPT Image 1 may return URL or b64_json depending on API version
            response = client.images.generate(
                model=self.model_name,
                prompt=prompt,
                size=self.size,
                quality=self.quality,  # "low", "medium", "high"
                n=1,
            )
            import base64
            from io import BytesIO
            from PIL import Image
            
            # Handle both b64_json and url formats
            result = response.data[0]
            if hasattr(result, 'b64_json') and result.b64_json:
                # Base64 encoded image
                image_base64 = result.b64_json
                image_data = base64.b64decode(image_base64)
                img = Image.open(BytesIO(image_data))
                return img
            elif hasattr(result, 'url') and result.url:
                # URL format - download the image
                import requests
                image_url = result.url
                img_response = requests.get(image_url)
                img = Image.open(BytesIO(img_response.content))
                return img
            else:
                raise ValueError("No image data found in response")
        elif self.model_name == "dall-e-3":
            response = client.images.generate(
                model=self.model_name,
                prompt=prompt,
                size=self.size,
                quality=self.quality,
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

