"""OpenRouter API model implementation for image generation."""

import os
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.models.base_model import ImageModel

logger = logging.getLogger(__name__)


class OpenRouterModel(ImageModel):
    """OpenRouter API model for image generation.
    
    OpenRouter provides access to various image generation models through
    an OpenAI-compatible API interface.
    """

    def __init__(
        self,
        name: str = "openrouter",
        model: str = "nano-banana/nano-banana",
        api_key: Optional[str] = None,
        size: str = "1024x1024",
        quality: str = "high",
    ) -> None:
        """Initialize OpenRouter model.

        Args:
            name: Model name.
            model: OpenRouter model identifier (e.g., "nano-banana/nano-banana").
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.
            size: Image size ("1024x1024", "1792x1024", "1024x1792").
            quality: Image quality (if supported by model).
        """
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your API key from https://openrouter.ai/"
            )

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
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"

    def generate(self, prompt: str, seed: Optional[int] = None) -> "Image.Image":
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the desired image.
            seed: Optional random seed (may not be supported, ignored).

        Returns:
            Generated PIL Image.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenRouter model requires openai package. Install with: pip install openai"
            ) from e

        # Initialize client with OpenRouter base URL
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        import base64
        import re
        from io import BytesIO
        from PIL import Image
        import requests
        
        # Gemini models use chat completions API for image generation
        # Try chat completions first (for Gemini models)
        # Check if this is a Gemini model
        is_gemini = 'gemini' in self.model_name.lower()
        
        if is_gemini:
            # Use chat completions for Gemini
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                )
                
                # Extract image from response
                message = response.choices[0].message
                
                # Check for images field (Gemini returns images here)
                if hasattr(message, 'images') and message.images:
                    for image_obj in message.images:
                        # Handle both dict and object formats
                        if isinstance(image_obj, dict):
                            image_url_obj = image_obj.get('image_url', {})
                            if isinstance(image_url_obj, dict):
                                image_url = image_url_obj.get('url', '')
                            else:
                                image_url = str(image_url_obj)
                        else:
                            if hasattr(image_obj, 'image_url'):
                                image_url_attr = image_obj.image_url
                                if hasattr(image_url_attr, 'url'):
                                    image_url = image_url_attr.url
                                else:
                                    image_url = str(image_url_attr)
                            else:
                                continue
                        
                        if not image_url:
                            continue
                        
                        # Handle base64 data URLs
                        if image_url.startswith('data:image'):
                            # Extract base64 data
                            base64_match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', image_url)
                            if base64_match:
                                image_data = base64.b64decode(base64_match.group(1))
                                img = Image.open(BytesIO(image_data))
                                return img
                        elif image_url.startswith('http'):
                            # Regular URL - download
                            img_response = requests.get(image_url)
                            img = Image.open(BytesIO(img_response.content))
                            return img
                
                # Check for content with embedded base64
                if hasattr(message, 'content') and message.content:
                    content = message.content
                    if isinstance(content, str):
                        # Try to extract base64 image from content
                        base64_match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', content)
                        if base64_match:
                            image_data = base64.b64decode(base64_match.group(1))
                            img = Image.open(BytesIO(image_data))
                            return img
                        
                        # Try to extract URL
                        url_match = re.search(r'https?://[^\s\)]+\.(?:png|jpg|jpeg|gif|webp)', content)
                        if url_match:
                            image_url = url_match.group(0)
                            img_response = requests.get(image_url)
                            img = Image.open(BytesIO(img_response.content))
                            return img
                
                # If no image found
                raise ValueError(f"Chat completions did not return image data in expected format")
                
            except Exception as e:
                logger.error(f"Chat completions failed: {e}")
                raise RuntimeError(f"Failed to generate image with OpenRouter: {e}") from e
        else:
            # Use images.generate for other models (DALL-E compatible)
            try:
                response = client.images.generate(
                    model=self.model_name,
                    prompt=prompt,
                    size=self.size,
                    n=1,
                )
                
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
                    image_url = result.url
                    img_response = requests.get(image_url)
                    img = Image.open(BytesIO(img_response.content))
                    return img
                else:
                    raise ValueError("No image data found in response")
                    
            except Exception as e:
                logger.error(f"OpenRouter API error: {e}")
                raise RuntimeError(f"Failed to generate image with OpenRouter: {e}") from e

