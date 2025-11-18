"""OCR backend abstraction for text extraction."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


class TextBackend(ABC):
    """Abstract OCR backend."""

    @abstractmethod
    def extract_text(self, image: "Image.Image") -> str:
        """Return raw text from the given image.

        Args:
            image: PIL Image to extract text from.

        Returns:
            Extracted text as a string.
        """
        raise NotImplementedError


class TesseractBackend(TextBackend):
    """Simple OCR backend using pytesseract."""

    def __init__(self, lang: str = "eng") -> None:
        """Initialize Tesseract backend.

        Args:
            lang: Language code for Tesseract (default: "eng").
        """
        self.lang = lang

    def extract_text(self, image: "Image.Image") -> str:
        """Extract text using pytesseract.

        Args:
            image: PIL Image to extract text from.

        Returns:
            Extracted text as a string.
        """
        import pytesseract

        return pytesseract.image_to_string(image, lang=self.lang)


class AdvancedBackend(TextBackend):
    """Placeholder for a more advanced OCR backend (e.g., Surya or DeepSeek-OCR)."""

    def __init__(self, model_name: str = "surya") -> None:
        """Initialize advanced backend placeholder.

        Args:
            model_name: Name of the advanced OCR model (default: "surya").
        """
        self.model_name = model_name

    def extract_text(self, image: "Image.Image") -> str:
        """Extract text using advanced OCR (placeholder).

        Args:
            image: PIL Image to extract text from.

        Returns:
            Extracted text as a string.

        Raises:
            NotImplementedError: Always, as this is a placeholder.
        """
        raise NotImplementedError(
            "AdvancedBackend is a placeholder. Integrate Surya / DeepSeek-OCR here."
        )


def build_text_backend(name: str) -> TextBackend:
    """Factory to select OCR backend by name.

    Args:
        name: Backend name ("tesseract", "default", "advanced", "surya", "deepseek").

    Returns:
        TextBackend instance.

    Raises:
        ValueError: If backend name is unknown.
    """
    name = name.lower()
    if name in ("tesseract", "default"):
        return TesseractBackend()
    elif name in ("advanced", "surya", "deepseek"):
        return AdvancedBackend(model_name=name)
    else:
        raise ValueError(f"Unknown OCR backend: {name}")

