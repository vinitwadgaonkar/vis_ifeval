"""Google Gemini (Nano Banana) API model implementation."""

import os
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.models.base_model import ImageModel

logger = logging.getLogger(__name__)


class GeminiModel(ImageModel):
    """Google Gemini (Nano Banana / Gemini 2.5 Flash Image) model for image generation."""

    def __init__(
        self,
        name: str = "gemini",
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp-image-generation",  # Free tier compatible
    ) -> None:
        """Initialize Gemini model.

        Args:
            name: Model name.
            api_key: Google API key. If None, reads from GEMINI_API_KEY or GOOGLE_API_KEY env var.
            model_name: Gemini model name (default: "gemini-2.5-flash-image").
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set. "
                "Get your API key from https://ai.google.dev/"
            )

        super().__init__(
            name=name,
            config={
                "model_name": model_name,
            },
        )
        self.api_key = api_key
        self.model_name = model_name
        self._load_client()

    def _load_client(self) -> None:
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(
                "Gemini model requires the 'google-generativeai' library. "
                "Install with: pip install google-generativeai"
            ) from e

    def generate(self, prompt: str, seed: Optional[int] = None) -> "Image.Image":
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the desired image.
            seed: Optional random seed (may not be supported by Gemini API, ignored).

        Returns:
            Generated PIL Image.
        """
        from io import BytesIO
        from PIL import Image

        try:
            # Initialize the model
            model = self.genai.GenerativeModel(model_name=self.model_name)

            # Generate content
            # Note: Gemini image generation requires billing enabled (free tier has 0 quota)
            import time
            max_retries = 3
            retry_delay = 3
            
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.7,
                        }
                    )
                    break  # Success, exit retry loop
                except Exception as api_error:
                    error_str = str(api_error)
                    if "quota" in error_str.lower() or "429" in error_str:
                        if attempt < max_retries - 1:
                            # Extract retry delay from error if available
                            if "retry in" in error_str.lower():
                                import re
                                delay_match = re.search(r'retry in ([\d.]+)s', error_str.lower())
                                if delay_match:
                                    retry_delay = float(delay_match.group(1)) + 1
                            logger.warning(
                                f"Quota limit hit, retrying in {retry_delay}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(retry_delay)
                            continue
                        else:
                            raise RuntimeError(
                                f"Gemini API quota exceeded after {max_retries} attempts. "
                                f"Image generation requires billing enabled on your Google Cloud account. "
                                f"Free tier has 0 quota for image generation. "
                                f"Please enable billing at https://ai.google.dev/ or wait for quota reset. "
                                f"Error: {api_error}"
                            ) from api_error
                    raise  # Re-raise if not a quota error

            # Gemini returns images in the response
            # The response structure may vary, so we handle different cases
            if hasattr(response, "images") and response.images:
                # If response has images attribute
                image_data = response.images[0]
                if isinstance(image_data, bytes):
                    img = Image.open(BytesIO(image_data))
                else:
                    # If it's a PIL Image already
                    img = image_data
            elif hasattr(response, "candidates") and response.candidates:
                # Check candidates for image data
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "inline_data"):
                            image_data = part.inline_data.data
                            img = Image.open(BytesIO(image_data))
                            break
                        elif hasattr(part, "image"):
                            img = part.image
                            break
                else:
                    raise ValueError("Unexpected response structure from Gemini API")
            elif hasattr(response, "text"):
                # If response has text, it might contain base64 image
                import base64
                import re
                # Try to extract base64 image from text
                base64_match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', response.text)
                if base64_match:
                    image_data = base64.b64decode(base64_match.group(1))
                    img = Image.open(BytesIO(image_data))
                else:
                    raise ValueError(f"Gemini API returned text instead of image: {response.text[:100]}")
            else:
                # Try to get image from response directly
                if hasattr(response, "image"):
                    img = response.image
                else:
                    raise ValueError(f"Unexpected response format: {dir(response)}")

            return img

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            # Log more details for debugging
            if hasattr(e, "response"):
                logger.error(f"Response: {e.response}")
            raise RuntimeError(f"Failed to generate image with Gemini: {e}") from e

