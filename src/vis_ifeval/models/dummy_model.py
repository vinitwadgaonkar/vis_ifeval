"""Dummy image model for testing."""

import numpy as np
from PIL import Image

from vis_ifeval.models.base_model import ImageModel


class DummyModel(ImageModel):
    """Dummy model that generates random images for testing."""

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the dummy model.

        Args:
            config: Optional configuration dictionary.
        """
        super().__init__(name="dummy", config=config)

    def generate(self, prompt: str, seed: int | None = None) -> Image.Image:
        """Generate a random 256x256 RGB image.

        Args:
            prompt: Text prompt (ignored for dummy model).
            seed: Optional random seed for reproducibility.

        Returns:
            Random 256x256 RGB PIL Image.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Generate random RGB image
        array = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        return Image.fromarray(array, mode="RGB")

