"""Logic evaluator for consistency checks."""

import logging
import math
import re
from typing import TYPE_CHECKING

from PIL import Image

from vis_ifeval.evaluators.base import ConstraintEvaluator
from vis_ifeval.evaluators.label_eval import LabelEvaluator
from vis_ifeval.utils.ocr_backend import TextBackend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LogicEvaluator(ConstraintEvaluator):
    """Evaluator for logic constraints."""

    def __init__(self, backend: TextBackend) -> None:
        """Initialize LogicEvaluator with an OCR backend.

        Args:
            backend: TextBackend instance for text extraction.
        """
        self.backend = backend
        # Reuse label parsing logic
        self._label_eval = LabelEvaluator(backend)

    def can_handle(self, constraint: dict) -> bool:
        """Check if constraint type is 'logic'."""
        return constraint.get("type") == "logic"

    def score(
        self, image: Image.Image, prompt: dict, constraint: dict
    ) -> float:
        """Score logic constraint.

        Currently supports:
        - percent_dv_consistency: Checks if sodium mg and %DV are consistent.

        Args:
            image: PIL Image to evaluate.
            prompt: Full prompt dictionary.
            constraint: Constraint dictionary.

        Returns:
            Score in [0, 1] indicating how well the logic constraint is satisfied.
        """
        logic_type = constraint.get("logic_type", "")
        if logic_type == "percent_dv_consistency":
            return self._check_percent_dv_consistency(image, constraint)
        else:
            logger.warning(f"Unknown logic_type: {logic_type}")
            return 0.0

    def _check_percent_dv_consistency(
        self, image: Image.Image, constraint: dict
    ) -> float:
        """Check if sodium mg and %DV are internally consistent.

        Uses a daily reference of 2300 mg for sodium.

        Args:
            image: PIL Image to evaluate.
            constraint: Constraint dictionary.

        Returns:
            Score in [0, 1] based on consistency.
        """
        try:
            # Reuse label parsing
            label_image = self._label_eval._crop_label_region(image)
            raw_text = self.backend.extract_text(label_image)
            parsed = self._label_eval._parse_fields(raw_text)

            if "sodium_mg" not in parsed or "sodium_dv_percent" not in parsed:
                logger.debug("Missing sodium_mg or sodium_dv_percent in parsed label")
                return 0.0

            # Parse values
            mg_str = parsed["sodium_mg"]
            dv_str = parsed["sodium_dv_percent"]

            mg_match = re.search(r"([0-9.]+)", mg_str)
            dv_match = re.search(r"([0-9.]+)", dv_str)

            if not mg_match or not dv_match:
                return 0.0

            mg = float(mg_match.group(1))
            dv = float(dv_match.group(1))

            # Daily reference for sodium (2300 mg)
            daily_reference = 2300.0
            implied_dv = 100.0 * mg / daily_reference

            # Compute relative error
            rel_err = abs(implied_dv - dv) / max(1.0, abs(dv))
            score = math.exp(-3.0 * rel_err)

            return max(0.0, min(1.0, float(score)))

        except Exception as e:
            logger.warning(f"Logic evaluation failed: {e}")
            return 0.0
