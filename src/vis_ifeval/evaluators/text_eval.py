"""Text-based constraint evaluator using OCR."""

import logging
import math

from Levenshtein import distance as levenshtein_distance
from PIL import Image

from vis_ifeval.evaluators.base import ConstraintEvaluator
from vis_ifeval.utils.ocr_backend import TextBackend

logger = logging.getLogger(__name__)


class TextEvaluator(ConstraintEvaluator):
    """Evaluator for text-based constraints using OCR."""

    def __init__(self, backend: TextBackend) -> None:
        """Initialize TextEvaluator with an OCR backend.

        Args:
            backend: TextBackend instance for text extraction.
        """
        self.backend = backend

    def can_handle(self, constraint: dict) -> bool:
        """Check if constraint type is 'text'."""
        return constraint.get("type") == "text"

    def score(
        self, image: Image.Image, prompt: dict, constraint: dict
    ) -> float:
        """Score text constraint using OCR and Character Error Rate.

        Args:
            image: PIL Image to evaluate.
            prompt: Full prompt dictionary.
            constraint: Constraint dictionary with 'target' field containing expected text.

        Returns:
            Score in [0, 1] based on Character Error Rate.
        """
        target_text = constraint.get("target", "")
        if not target_text:
            logger.warning("Text constraint missing 'target' field")
            return 0.0

        # Handle region hints
        region = constraint.get("region")
        if region == "label_top":
            # Crop top 30% of image
            width, height = image.size
            crop_box = (0, 0, width, int(height * 0.3))
            ocr_image = image.crop(crop_box)
        else:
            ocr_image = image

        try:
            # Perform OCR using backend
            ocr_text = self.backend.extract_text(ocr_image).strip()
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return 0.0

        # Normalize text (uppercase, remove extra whitespace)
        ocr_normalized = " ".join(ocr_text.upper().split())
        target_normalized = " ".join(target_text.upper().split())

        # Compute Character Error Rate (CER)
        if not target_normalized:
            return 1.0 if not ocr_normalized else 0.0

        edit_dist = levenshtein_distance(ocr_normalized, target_normalized)
        max_len = max(len(target_normalized), 1)
        cer = edit_dist / max_len

        # Convert CER to score using exponential decay
        alpha = 3.0
        score = math.exp(-alpha * cer)
        return max(0.0, min(1.0, score))

