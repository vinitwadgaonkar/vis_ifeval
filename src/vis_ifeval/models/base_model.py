"""Base interface for image generation models."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


class ImageModel(ABC):
    """Abstract base class for image generation models."""

    def __init__(self, name: str, config: dict | None = None) -> None:
        """Initialize the model.

        Args:
            name: Name of the model.
            config: Optional configuration dictionary.
        """
        self.name = name
        self.config = config or {}

    @abstractmethod
    def generate(self, prompt: str, seed: int | None = None) -> "Image.Image":
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the desired image.
            seed: Optional random seed for reproducibility.

        Returns:
            Generated PIL Image.
        """
        pass

